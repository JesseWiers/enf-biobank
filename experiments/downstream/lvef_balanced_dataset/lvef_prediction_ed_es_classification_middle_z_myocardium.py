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
import traceback
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# Custom imports
from experiments.datasets.biobank_latent_dataset_myocardium import LatentEndpointDatasetEDESMiddleSlicesMyocardium, collate_fn
from enf.model import EquivariantNeuralField
from experiments.downstream_models.ponita import PonitaFixedSize
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier
from experiments.downstream_models.egnn import EGNNClassifier

from experiments.datasets.biobank_latent_dataset_myocardium import create_dataloaders


jax.config.update("jax_default_matmul_precision", "highest")


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    
    # Fixed seed for dataset splitting - this will always be the same
    config.dataset_seed = 42
    # Separate seed for model initialization - change this for different model initializations
    config.model_seed = 42  # Different from dataset seed

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.latent_path = '/projects/prjs1252/data_jesse_v2/latent_dataset_lvef_32l_16d_psnr_24.h5'
    config.dataset.csv_path = '/projects/prjs1252/data_jesse_v2/metadata/filtered_endpoints.csv'
    config.dataset.debug_limit = None
    config.dataset.z_indices = (0, 2, 4)
    config.dataset.k_nearest = 16  # Number of points to consider
    config.dataset.exclude_k_nearest = False  # Add this line to control whether to exclude k nearest points
    config.dataset.reset_context_vectors = True
    config.dataset.reset_es_timepoint = True  # Add this line
    config.dataset.normalize_context_vectors = True  # Add this for consistency

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_model = 1e-4 # 0.0001
    
    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "TransformerClassifier"  # Can be "TransformerClassifier", "ponita", or "egnn"
    config.model.hidden_size = 768
    config.model.num_classes = 2

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train_cls = 10
    config.train.validation_interval = 10
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.exp_name = "test"
    config.run_name = "lvef_prediction"
    
    # Add EGNN-specific config
    config.model.egnn = ml_collections.ConfigDict()
    config.model.egnn.k_neighbors = 8
    config.model.egnn.num_layers = 12
    config.model.egnn.use_radius = False  # alternative to k-neighbors
    config.model.egnn.radius = 1.0  # only used if use_radius is True
    
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online", name=config.run_name)

    # Load dataset with three-way split using dataset seed
    train_dloader, val_dloader, test_dloader = create_dataloaders(
        hdf5_path=config.dataset.latent_path,
        endpoints_csv_path=config.dataset.csv_path,
        batch_size=config.train.batch_size,
        train_split=0.7,  # 70% training
        val_split=0.15,   # 15% validation
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit,
        k_nearest=config.dataset.k_nearest,
        exclude_k_nearest=config.dataset.exclude_k_nearest,
        random_seed=config.dataset_seed  # Use dataset seed for consistent splits
    )
    
    logging.info(f"Train dataloader length: {len(train_dloader)}")
    logging.info(f"Val dataloader length: {len(val_dloader)}")
    logging.info(f"Test dataloader length: {len(test_dloader)}")

    # Load sample batch
    sample_batch = next(iter(train_dloader))
    patient_id, z, endpoint_value = sample_batch
    
    # Use model seed for initialization
    key = jax.random.PRNGKey(config.model_seed)  # Use model seed for model initialization

    # Create dummy latents for model init
    key, subkey = jax.random.split(key)
    
    # Obtain dataset metadata
    with h5py.File(config.dataset.latent_path, 'r') as f:
        num_latents_slice = f['metadata'].attrs['num_latents']
        latent_dim = f['metadata'].attrs['latent_dim']

    temp_z = initialize_latents(
        batch_size=1,  # Only need one example for initialization
        num_latents=num_latents_slice*2, 
        latent_dim=latent_dim,
        data_dim=4,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.train.noise_scale,
    )

    # Define the model
    if config.model.name == "TransformerClassifier":
        model = TransformerClassifier(
            hidden_size=config.model.hidden_size,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            num_classes=config.model.num_classes,
        )
        model_params = model.init(key, *temp_z)
    elif config.model.name == "ponita":
        model = PonitaFixedSize(
            num_hidden=128,
            num_layers=4,
            scalar_num_out=1,
            vec_num_out=0,
            spatial_dim=2,
            num_ori=1,
            basis_dim=128,
            degree=3,
            widening_factor=4,
            global_pool=True,
            kernel_size=0.25,
            last_feature_conditioning=False,
        )
        model_params = model.init(key, temp_z)
    elif config.model.name == "egnn":
        model = EGNNClassifier(
            hidden_dim=config.model.hidden_size,
            num_layers=12,
            num_classes=config.model.num_classes,
            k_neighbors=config.model.egnn.k_neighbors,
        )
        p, c, g = temp_z  # Unpack for EGNN
        model_params = model.init(key, p, c, g)
    else:
        raise ValueError(f"Model {config.model.name} not found")
    
    # Define optimizer for the model model
    model_opt = optax.adam(learning_rate=config.optim.lr_model)
    model_opt_state = model_opt.init(model_params)
    
    # Compute mean and std of the context vectors for normalization for faster convergence
    c_list = []
    for i, (_, z_tuples, _) in enumerate(train_dloader):
        c_list.append(z_tuples[1])
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    
    @jax.jit
    def classifier_step(z, targets, model_params, model_opt_state, key):
        def cross_entropy_loss(params):
            # Unpack the tuple
            p, c, g = z
            
            # Normalize the context vectors
            c = (c - c_mean) / c_std
            
            # Convert targets to binary labels (0 if < 40, 1 if >= 40)
            binary_targets = (targets >= 40.0).astype(jnp.int32)
            
            # Forward pass through model
            if config.model.name == "TransformerClassifier":
                logits = model.apply(params, p, c, g)
            elif config.model.name == "ponita":
                logits = model.apply(params, (p, c, g))
            elif config.model.name == "egnn":
                logits = model.apply(params, p, c, g)
            
            # Compute cross entropy loss
            labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
            loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
                
            return loss

        # Get gradients
        loss, grads = jax.value_and_grad(cross_entropy_loss)(model_params)

        # Update model parameters
        updates, model_opt_state = model_opt.update(grads, model_opt_state)
        model_params = optax.apply_updates(model_params, updates)

        return loss, model_params, model_opt_state
    

    # Training loop
    global_step = 0
    best_val_accuracy = 0.0
    best_model_params = None

    for epoch in range(config.train.num_epochs_train_cls):
        epoch_losses = []
        
        # Training phase
        for batch in tqdm(train_dloader, desc=f"Training Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            p, c, g = z
            
            # Add ES timepoint reset
            if config.dataset.reset_es_timepoint:
                # Set time coordinate of ES points (second half) to 1
                num_points_per_timepoint = p.shape[1] // 2
                p = p.at[:, num_points_per_timepoint:, 0].set(1)

            z = (p, c, g)
            
            # Training step
            loss, model_params, model_opt_state = classifier_step(
                z, targets, model_params, model_opt_state, key
            )
            
            epoch_losses.append(loss)
            global_step += 1  # Increment global step
            
        # Calculate and log average training loss for the epoch
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        logging.info(f"Epoch {epoch}: train-Loss = {avg_loss:.4f}")
        wandb.log({
            "train/loss": avg_loss,
            "train/step": global_step,
            "train/epoch": epoch
        }, step=global_step)  # Use global_step instead of epoch
        epoch_losses = []  # Reset loss tracking
        
        # Validation phase
        val_losses = []
        val_accuracies = []
        misclassified_data = []  # New list to store misclassified information
        
        for batch in tqdm(val_dloader, desc=f"Validation Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            p, c, g = z
            
            # Add ES timepoint reset
            if config.dataset.reset_es_timepoint:
                # Set time coordinate of ES points (second half) to 1
                num_points_per_timepoint = p.shape[1] // 2  # Calculate based on actual points
                p = p.at[:, num_points_per_timepoint:, 0].set(1)

            z = (p, c, g)
            
            # Forward pass through model
            if config.model.name == "TransformerClassifier":
                logits = model.apply(model_params, p, c, g)
            elif config.model.name == "ponita":
                logits = model.apply(model_params, (p, c, g))
            elif config.model.name == "egnn":
                logits = model.apply(model_params, p, c, g)
            
            binary_targets = (targets >= 40.0).astype(jnp.int32)
            
            # Compute loss and accuracy
            labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
            loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
            
            # Track misclassified patients
            misclassified_mask = preds != binary_targets
            if jnp.any(misclassified_mask):
                misclassified_indices = jnp.where(misclassified_mask)[0]
                for idx in misclassified_indices:
                    misclassified_data.append({
                        'patient_id': patient_ids[idx],
                        'true_lvef': float(targets[idx]),
                        'predicted_class': int(preds[idx]),
                        'true_class': int(binary_targets[idx])
                    })
            
            val_losses.append(loss)
            val_accuracies.append(accuracy)
            
        # Log validation metrics
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        avg_val_accuracy = jnp.mean(jnp.array(val_accuracies))
        logging.info(f"Epoch {epoch}: val-Loss = {avg_val_loss:.4f}, val-Accuracy = {avg_val_accuracy:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_accuracy,
            "epoch": epoch
        })
        
        # Log misclassified patients
        if misclassified_data:
            # Create a table for misclassified patients
            misclassified_table = wandb.Table(
                columns=["patient_id", "true_lvef", "predicted_class", "true_class"]
            )
            for item in misclassified_data:
                misclassified_table.add_data(
                    item['patient_id'],
                    item['true_lvef'],
                    "LVEF ≥ 40" if item['predicted_class'] == 1 else "LVEF < 40",
                    "LVEF ≥ 40" if item['true_class'] == 1 else "LVEF < 40"
                )
            
            wandb.log({f"val_errors/misclassified_patients_epoch_{epoch}": misclassified_table})
            
            # Also log to console
            logging.info(f"\nMisclassified patients in epoch {epoch}:")
            for item in misclassified_data:
                logging.info(
                    f"Patient {item['patient_id']}: "
                    f"True LVEF = {item['true_lvef']:.1f}, "
                    f"Predicted = {'LVEF ≥ 40' if item['predicted_class'] == 1 else 'LVEF < 40'}, "
                    f"Actual = {'LVEF ≥ 40' if item['true_class'] == 1 else 'LVEF < 40'}"
                )

        # Save best model based on validation accuracy
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_model_params = jax.tree_map(lambda x: x.copy(), model_params)
            logging.info(f"New best validation accuracy: {best_val_accuracy:.4f}")

    # Final evaluation on test set using best model
    logging.info("\nPerforming final evaluation on test set using best model...")
    test_losses = []
    test_accuracies = []
    test_predictions = []
    test_true_values = []
    
    for batch in tqdm(test_dloader, desc="Test set evaluation"):
        patient_ids, z, targets = batch
        p, c, g = z
        
        if config.dataset.reset_es_timepoint:
            num_points_per_timepoint = p.shape[1] // 2
            p = p.at[:, num_points_per_timepoint:, 0].set(1)
        
        # Normalize context vectors
        c = (c - c_mean) / c_std
        
        # Forward pass using best model
        if config.model.name == "TransformerClassifier":
            logits = model.apply(best_model_params, p, c, g)
        elif config.model.name == "ponita":
            logits = model.apply(best_model_params, (p, c, g))
        elif config.model.name == "egnn":
            logits = model.apply(best_model_params, p, c, g)
            
        binary_targets = (targets >= 40.0).astype(jnp.int32)
        
        # Compute metrics
        labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
        loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
        
        test_losses.append(loss)
        test_accuracies.append(accuracy)
        test_predictions.extend(preds)
        test_true_values.extend(binary_targets)
    
    # Compute and log final test metrics
    final_test_loss = jnp.mean(jnp.array(test_losses))
    final_test_accuracy = jnp.mean(jnp.array(test_accuracies))
    
    # Calculate additional metrics
    test_predictions = jnp.array(test_predictions)
    test_true_values = jnp.array(test_true_values)
    
    tp = jnp.sum((test_predictions == 1) & (test_true_values == 1))
    fp = jnp.sum((test_predictions == 1) & (test_true_values == 0))
    fn = jnp.sum((test_predictions == 0) & (test_true_values == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Log final results
    logging.info("\nFinal Test Results:")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Test Loss: {final_test_loss:.4f}")
    logging.info(f"Test Accuracy: {final_test_accuracy:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1 Score: {f1:.4f}")
    
    # Log to wandb
    wandb.log({
        "test/final_loss": final_test_loss,
        "test/final_accuracy": final_test_accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
        "best_validation_accuracy": best_val_accuracy
    })

    run.finish()


if __name__ == "__main__":
    app.run(main)
