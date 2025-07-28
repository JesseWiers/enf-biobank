import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import h5py
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb

# Custom imports
from experiments.datasets.biobank_latent_endpoint_dataset_autodecoding import create_dataloaders

from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier
from experiments.downstream_models.egnn import EGNNClassifier

import numpy as np  # For numpy operations
import sklearn.metrics  # For AUROC computation
import time  # Add at the top if not already imported

jax.config.update("jax_default_matmul_precision", "highest")


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    
    # Fixed seed for dataset splitting - this will always be the same
    config.dataset_seed = 42
    # Separate seed for model initialization - change this for different model initializations
    config.model_seed = 42  # You can change this value to try different model initializations
    
    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    # Paths for healthy and endpoint latent datasets
    config.dataset.healthy_latent_path = '/projects/prjs1252/data_jesse_final_v3/autodecoding_latent_datasets/4096latents_32dim_healthy_matched_final_endpoint_500epochs_900_healthy_pretrain_1000patients_3200_epochs_900patients.h5'
    config.dataset.endpoint = 'cardiomyopathy'  
    config.dataset.num_workers = 0
    config.dataset.debug_limit = None
    
    # Latent structure config
    config.dataset.latents_per_dim = 8  # 8x8x8x8 = 4096 latents
    config.dataset.z_indices = 3  # Will select middle 3 z-slices
    config.dataset.t_indices = 2  # Will select middle 2 timepoints

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_model = 1e-4
    
    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "transformer"  # Can be "transformer" or "egnn"
    config.model.hidden_size = 768
    config.model.num_classes = 2  # Binary classification
    config.model.transformer_depth = 12
    config.model.num_heads = 12
    config.model.mlp_ratio = 4
    
    # Add EGNN-specific config
    config.model.egnn = ml_collections.ConfigDict()
    config.model.egnn.k_neighbors = 8
    config.model.egnn.num_layers = 12
    config.model.egnn.use_radius = False
    config.model.egnn.radius = 1.0

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 32
    config.train.noise_scale = 1e-1
    config.train.num_epochs_train_cls = 10  # Changed from num_epochs_train to num_epochs_train_cls
    config.train.val_split = 0.2
    config.train.val_frequency = None  # None for per-epoch validation, or int N for validation every N steps
    
    # Wandb config
    config.exp_name = "endpoint_classifications_autodecoding_tests"
    config.run_name = "cardiomyopathy"  # Will be set to endpoint name if empty
    
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def compute_context_stats(dataloader):
    """Compute mean and std of context vectors in a memory-efficient way"""
    sum_c = None
    total_points = 0
    
    for _, z_tuples, _ in tqdm(dataloader, desc="Computing context statistics mean"):
        c = z_tuples[1]
        batch_sum = jnp.sum(c, axis=(0, 1))
        if sum_c is None:
            sum_c = batch_sum
        else:
            sum_c += batch_sum
        total_points += c.shape[0] * c.shape[1]
    
    mean_c = sum_c / total_points
    
    # Second pass for std
    sum_sq_diff = None
    for _, z_tuples, _ in tqdm(dataloader, desc="Computing context statistics std"):
        c = z_tuples[1]
        diff = c - mean_c
        batch_sum_sq = jnp.sum(diff * diff, axis=(0, 1))
        if sum_sq_diff is None:
            sum_sq_diff = batch_sum_sq
        else:
            sum_sq_diff += batch_sum_sq
    
    std_c = jnp.sqrt(sum_sq_diff / total_points)
    std_c = jnp.where(std_c == 0, 1.0, std_c)  # Prevent division by zero
    return mean_c, std_c


def compute_metrics(logits, targets):
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean((preds == targets).astype(jnp.float32))
    
    # Compute precision, recall, F1
    tp = jnp.sum((preds == 1) & (targets == 1))
    fp = jnp.sum((preds == 1) & (targets == 0))
    fn = jnp.sum((preds == 0) & (targets == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main(_):
    # Get config
    config = _CONFIG.value
    
    # Set run name to endpoint if not specified
    if not config.run_name:
        config.run_name = f"{config.dataset.endpoint}_classification"

    # Initialize wandb
    run = wandb.init(
        project=config.exp_name, 
        config=config.to_dict(), 
        mode="online", 
        name=config.run_name
    )

    # Load dataset with three-way split using new dataset class
    train_dloader, val_dloader, test_dloader = create_dataloaders(
        healthy_hdf5_path=config.dataset.healthy_latent_path,
        endpoint_name=config.dataset.endpoint,
        batch_size=config.train.batch_size,
        train_split=0.75,
        val_split=0.15,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit,
        random_seed=config.dataset_seed,
        z_indices=config.dataset.z_indices,
        t_indices=config.dataset.t_indices,
    )
    
    logging.info(f"Train dataloader length: {len(train_dloader)}")
    logging.info(f"Val dataloader length: {len(val_dloader)}")
    logging.info(f"Test dataloader length: {len(test_dloader)}")

    # Use separate seed for model initialization
    key = jax.random.PRNGKey(config.model_seed)  # This will use whatever model_seed you set

    # Get sample batch for initialization
    sample_batch = next(iter(train_dloader))
    _, (p, c, g), _ = sample_batch  # Unpack the tuple of latents
    
    # Define the model
    if config.model.name == "transformer":
        model = TransformerClassifier(
            hidden_size=config.model.hidden_size,
            depth=config.model.transformer_depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            num_classes=config.model.num_classes,
        )
        model_params = model.init(key, p, c, g)  # Pass all three components
    else:
        raise ValueError("Only transformer model supported for autodecoding latents")

    # Define optimizer
    model_opt = optax.adam(learning_rate=config.optim.lr_model)
    model_opt_state = model_opt.init(model_params)
    
    # Compute mean and std of the context vectors for normalization
    c_list = []
    for _, z_tuple, _ in train_dloader:  # z_tuple is already (p, c, g)
        _, c, _ = z_tuple  # Unpack the tuple to get just c
        c_list.append(c)
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    
    @jax.jit
    def classifier_step(z, targets, model_params, model_opt_state, key):
        def cross_entropy_loss(params):
            # Unpack the tuple
            p, c, g = z
            
            # Normalize only the context vectors
            c = (c - c_mean) / c_std
            
            # Forward pass through model
            logits = model.apply(params, p, c, g)
            
            # Compute cross entropy loss
            labels_onehot = jax.nn.one_hot(targets, num_classes=2)
            loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
            return loss

        loss, grads = jax.value_and_grad(cross_entropy_loss)(model_params)
        updates, model_opt_state = model_opt.update(grads, model_opt_state)
        model_params = optax.apply_updates(model_params, updates)

        return loss, model_params, model_opt_state

    # Training loop
    global_step = 0
    best_val_accuracy = 0.0
    best_model_params = None

    for epoch in range(config.train.num_epochs_train_cls):
        epoch_losses = []
        epoch_accuracies = []
        
        # Training phase
        train_start_time = time.time()
        for batch in tqdm(train_dloader, desc=f"Training Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            # Training step
            loss, model_params, model_opt_state = classifier_step(
                z, targets, model_params, model_opt_state, key
            )
            
            # Compute accuracy
            p, c, g = z
            c = (c - c_mean) / c_std
            logits = model.apply(model_params, p, c, g)
            preds = jnp.argmax(logits, axis=-1)
            binary_targets = targets
            accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
            
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            global_step += 1
        train_duration = time.time() - train_start_time
        logging.info(f"Epoch {epoch} training took {train_duration:.2f} seconds")
        
        # Calculate and log average training metrics
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        avg_train_accuracy = jnp.mean(jnp.array(epoch_accuracies))
        logging.info(f"Epoch {epoch}: train-Loss = {avg_train_loss:.4f}, train-Accuracy = {avg_train_accuracy:.4f}")
        # Training metrics - use global_step
        wandb.log({
            "train/loss": avg_train_loss,
            "train/accuracy": avg_train_accuracy,
            "train/step": global_step,
            "train/epoch": epoch
        }, step=epoch)
        
        # Validation phase
        val_start_time = time.time()
        val_losses = []
        val_accuracies = []
        num_val_samples = 0
        
        for val_batch in tqdm(val_dloader, desc=f"Validation epoch {epoch}"):
            
            patient_ids, latents, targets = val_batch
            p, c, g = latents
            c = (c - c_mean) / c_std
            
            # Forward pass
            logits = model.apply(model_params, p, c, g)
            
            # Convert targets to binary (like in training)
            binary_targets = targets
            
            # Compute metrics
            labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
            loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
            
            val_losses.append(loss)
            val_accuracies.append(accuracy)
            num_val_samples += len(patient_ids)
        val_duration = time.time() - val_start_time
        logging.info(f"Epoch {epoch} validation took {val_duration:.2f} seconds")
        
        # Total epoch time
        epoch_duration = train_duration + val_duration
        logging.info(f"Epoch {epoch} total time: {epoch_duration:.2f} seconds ({train_duration:.2f}s training, {val_duration:.2f}s validation)")
        
        # Log validation metrics
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        avg_val_accuracy = jnp.mean(jnp.array(val_accuracies))
        
        # Save best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_model_params = jax.tree_map(lambda x: x.copy(), model_params)  # Deep copy of model parameters
            logging.info(f"New best validation accuracy: {avg_val_accuracy:.4f}")
        
        logging.info(f"Epoch {epoch} validation ({val_duration:.2f}s): Loss = {avg_val_loss:.4f}, Accuracy = {avg_val_accuracy:.4f}")
        # Validation metrics - use epoch (default)
        wandb.log({
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_accuracy,
            "epoch": epoch
        })  # No step parameter, will use default epoch-based stepping

    # Load best model for final evaluation
    logging.info(f"Loading best model (validation accuracy: {best_val_accuracy:.4f}) for test evaluation")
    model_params = best_model_params  # Use best model parameters
    
    # Final evaluation on test set using best model
    logging.info("\nPerforming final evaluation on test set using best model...")
    test_losses = []
    test_accuracies = []
    test_predictions = []
    test_true_values = []
    test_metrics = []  # Add this to collect metrics
    for batch in tqdm(test_dloader, desc="Test set evaluation"):
        patient_ids, z, targets = batch
        p, c, g = z
        c = (c - c_mean) / c_std
        
        # Forward pass
        logits = model.apply(best_model_params, p, c, g)
        
        # Convert targets to binary
        binary_targets = targets
        
        # Compute metrics
        labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
        loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
        test_losses.append(loss)
        batch_metrics = compute_metrics(logits, binary_targets)
        test_metrics.append(batch_metrics)
        
        test_accuracies.append(batch_metrics['accuracy'])

    # Compute average metrics correctly
    avg_test_metrics = {
        'accuracy': jnp.mean(jnp.array(test_accuracies)),
        'precision': jnp.mean(jnp.array([m['precision'] for m in test_metrics])),
        'recall': jnp.mean(jnp.array([m['recall'] for m in test_metrics])),
        'f1': jnp.mean(jnp.array([m['f1'] for m in test_metrics]))
    }
    
    # Log final test results
    logging.info("Final Test Results (using best validation model):")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    avg_test_loss = jnp.mean(jnp.array(test_losses))
    logging.info(f"Test Loss: {avg_test_loss:.4f}")
    logging.info(f"Test Accuracy: {avg_test_metrics['accuracy']:.4f}")
    logging.info(f"Test Precision: {avg_test_metrics['precision']:.4f}")
    logging.info(f"Test Recall: {avg_test_metrics['recall']:.4f}")
    logging.info(f"Test F1 Score: {avg_test_metrics['f1']:.4f}")
    
    # Log to wandb
    # Test metrics - single point at end of training
    wandb.log({
        "test/loss": avg_test_loss,
        "test/accuracy": avg_test_metrics['accuracy'],
        "test/precision": avg_test_metrics['precision'],
        "test/recall": avg_test_metrics['recall'],
        "test/f1": avg_test_metrics['f1'],
        "best_validation_accuracy": best_val_accuracy
    })  # No step parameter for final test metrics

    run.finish()

if __name__ == "__main__":
    app.run(main)
