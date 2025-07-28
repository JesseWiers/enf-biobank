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
from torch.utils.data import DataLoader, SubsetRandomSampler

# Custom imports
from experiments.datasets.biobank_latent_dataset import LatentEndpointDatasetEDESMiddleSlices, LatentEndpointDatasetEDESMiddleSlicesRotated, collate_fn
from enf.model import EquivariantNeuralField
from experiments.downstream_models.ponita import PonitaFixedSize
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

jax.config.update("jax_default_matmul_precision", "highest")


def create_split_indices(dataset_size, train_split=0.7, val_split=0.15, random_seed=42):
    """
    Create deterministic train/val/test split indices.
    
    Args:
        dataset_size (int): Total number of samples
        train_split (float): Proportion for training (default: 0.7)
        val_split (float): Proportion for validation (default: 0.15)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    np.random.seed(random_seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def create_dataloaders(hdf5_path, endpoints_csv_path, batch_size=16, endpoint_name='LVEF', 
                      train_split=0.7, val_split=0.15, random_seed=42, 
                      num_workers=0, debug_limit=None):
    """
    Create train, validation and test dataloaders.
    Train and validation use original latents, test uses rotated versions.
    
    Splits:
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create datasets
    try:
        train_val_dataset = LatentEndpointDatasetEDESMiddleSlices(
            hdf5_path, 
            endpoints_csv_path, 
            endpoint_name,
            debug_limit=debug_limit
        )
        
        test_dataset = LatentEndpointDatasetEDESMiddleSlicesRotated(
            hdf5_path,
            endpoints_csv_path,
            endpoint_name,
            debug_limit=debug_limit
        )
        
    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        print(traceback.format_exc())
        raise
    
    # Get dataset size and create split indices
    dataset_size = len(train_val_dataset)
    train_indices, val_indices, test_indices = create_split_indices(
        dataset_size, train_split, val_split, random_seed
    )
    
    # Create dataloaders with the same indices
    train_loader = DataLoader(
        train_val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    val_loader = DataLoader(
        train_val_dataset,  # Use original dataset for validation
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        test_dataset,  # Use rotated dataset for testing
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    print(f"Created dataloaders with:")
    print(f"- Training: {len(train_indices)} samples")
    print(f"- Validation: {len(val_indices)} samples")
    print(f"- Test: {len(test_indices)} samples (using rotated versions)")
    
    return train_loader, val_loader, test_loader


def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.latent_path = '/projects/prjs1252/data_jesse_v2/latent_dataset_lvef_32l_16d_psnr_24_rotations.h5'
    config.dataset.csv_path = '/projects/prjs1252/data_jesse_v2/metadata/filtered_endpoints.csv'
    config.dataset.z_indices = (3, 4, 5, 6)
    config.dataset.debug_limit = None

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_model = 0.000001 # 0.0001
    
    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "TransformerClassifier"
    config.model.hidden_size = 768
    config.model.num_classes = 2

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train_cls = 1
    config.train.validation_interval = 50
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.exp_name = "test"
    config.run_name = "lvef_prediction"
    
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, val_dloader, test_dloader = create_dataloaders(
        hdf5_path=config.dataset.latent_path,
        endpoints_csv_path=config.dataset.csv_path,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit
    )
    
    logging.info(f"Train dataloader length: {len(train_dloader)}")
    logging.info(f"Val dataloader length: {len(val_dloader)}")
    logging.info(f"Test dataloader length: {len(test_dloader)}")

    # Load sample batch
    sample_batch = next(iter(train_dloader))
    patient_id, z, endpoint_value = sample_batch
    
    # Random key
    key = jax.random.PRNGKey(55)

    # Create dummy latents for model init
    key, subkey = jax.random.split(key)
    
    # Obtain dataset metadata
    with h5py.File(config.dataset.latent_path, 'r') as f:
        num_latents_slice = f['metadata'].attrs['num_latents']
        latent_dim = f['metadata'].attrs['latent_dim']

    temp_z = initialize_latents(
        batch_size=1,  # Only need one example for initialization
        num_latents=num_latents_slice*2*len(config.dataset.z_indices), 
        latent_dim=latent_dim,
        data_dim=4,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.train.noise_scale,
    )

    # Define the  model
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
    

    # Add variables to track best model
    best_val_accuracy = 0.0
    best_model_params = None
    
    # Training loop
    global_step = 0
    for epoch in range(config.train.num_epochs_train_cls):
        # Log the epoch number to wandb at the start of each epoch
        wandb.log({"epoch": epoch}, step=global_step)
        logging.info(f"\n=== Starting Epoch {epoch} ===")
        
        epoch_losses = []
        
        # Add tqdm progress bar
        for batch in tqdm(train_dloader, desc=f"Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            # Training step
            loss, model_params, model_opt_state = classifier_step(
                z, targets, model_params, model_opt_state, key
            )
            
            epoch_losses.append(loss)
            global_step += 1

            # Log training metrics
            if global_step % config.train.validation_interval == 0:
                # Calculate and log average training loss
                avg_loss = jnp.mean(jnp.array(epoch_losses))
                logging.info(f"Step {global_step}: train-Loss = {avg_loss:.4f}")
                wandb.log({"train-Loss": avg_loss}, step=global_step)
                epoch_losses = []  # Reset loss tracking
                
                # Validation loop
                val_losses = []
                val_accuracies = []
                misclassified_data = []  # New list to store misclassified information
                
                for batch in tqdm(val_dloader, desc=f"Validation at step {global_step}"):
                    patient_ids, z, targets = batch
                    
                    # Unpack the tuple
                    p, c, g = z
                    
                    # Normalize the context vectors
                    c = (c - c_mean) / c_std
                    
                    # Forward pass through model
                    if config.model.name == "TransformerClassifier":
                        logits = model.apply(model_params, p, c, g)
                    elif config.model.name == "ponita":
                        logits = model.apply(model_params, (p, c, g))
                    
                    binary_targets = (targets >= 40.0).astype(jnp.int32)
                    
                    # Compute loss and accuracy
                    labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
                    loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
                    preds = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
                    
                    # Track misclassified patients
                    misclassified_mask = preds != binary_targets
                    if jnp.any(misclassified_mask):
                        # Convert JAX arrays to numpy for easier handling
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
                logging.info(f"Step {global_step}: val-Loss = {avg_val_loss}, val-Accuracy = {avg_val_accuracy}")
                
                # Log standard metrics
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/accuracy": avg_val_accuracy
                }, step=global_step)
                
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
                    
                    wandb.log({f"val_errors/misclassified_patients_step_{global_step}": misclassified_table}, step=global_step)
                    
                    # Also log to console
                    logging.info(f"\nMisclassified patients at step {global_step}:")
                    for item in misclassified_data:
                        logging.info(
                            f"Patient {item['patient_id']}: "
                            f"True LVEF = {item['true_lvef']:.1f}, "
                            f"Predicted = {'LVEF ≥ 40' if item['predicted_class'] == 1 else 'LVEF < 40'}, "
                            f"Actual = {'LVEF ≥ 40' if item['true_class'] == 1 else 'LVEF < 40'}"
                        )
                
                if avg_val_accuracy > best_val_accuracy:
                    best_val_accuracy = avg_val_accuracy
                    best_model_params = model_params
                    logging.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
    
    # Final evaluation on test set using best model
    logging.info("\n=== Final Test Set Evaluation ===")
    test_losses = []
    test_accuracies = []
    misclassified_data = []
    
    for batch in tqdm(test_dloader, desc="Test Set Evaluation"):
        patient_ids, z, targets = batch
        
        # Unpack the tuple
        p, c, g = z
        
        # Normalize the context vectors
        c = (c - c_mean) / c_std
        
        # Forward pass through model using best parameters
        if config.model.name == "TransformerClassifier":
            logits = model.apply(best_model_params, p, c, g)
        elif config.model.name == "ponita":
            logits = model.apply(best_model_params, (p, c, g))
        
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
        
        test_losses.append(loss)
        test_accuracies.append(accuracy)
    
    # Compute and log final test metrics
    final_test_loss = jnp.mean(jnp.array(test_losses))
    final_test_accuracy = jnp.mean(jnp.array(test_accuracies))
    
    logging.info(f"\nFinal Test Results:")
    logging.info(f"Test Loss: {final_test_loss:.4f}")
    logging.info(f"Test Accuracy: {final_test_accuracy:.4f}")
    
    # Log to wandb
    wandb.log({
        "test/final_loss": final_test_loss,
        "test/final_accuracy": final_test_accuracy
    })
    
    # Log misclassified test cases
    if misclassified_data:
        test_errors_table = wandb.Table(
            columns=["patient_id", "true_lvef", "predicted_class", "true_class"]
        )
        for item in misclassified_data:
            test_errors_table.add_data(
                item['patient_id'],
                item['true_lvef'],
                "LVEF ≥ 40" if item['predicted_class'] == 1 else "LVEF < 40",
                "LVEF ≥ 40" if item['true_class'] == 1 else "LVEF < 40"
            )
        
        wandb.log({"test/misclassified_patients": test_errors_table})
        
        # Log to console
        logging.info("\nMisclassified test patients:")
        for item in misclassified_data:
            logging.info(
                f"Patient {item['patient_id']}: "
                f"True LVEF = {item['true_lvef']:.1f}, "
                f"Predicted = {'LVEF ≥ 40' if item['predicted_class'] == 1 else 'LVEF < 40'}, "
                f"Actual = {'LVEF ≥ 40' if item['true_class'] == 1 else 'LVEF < 40'}"
            )
    
    run.finish()


if __name__ == "__main__":
    app.run(main)
