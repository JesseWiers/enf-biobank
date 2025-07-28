import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import h5py
import logging
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

# Custom imports
from experiments.datasets.biobank_latent_dataset import LatentEndpointDatasetAutodecoding, collate_fn
from experiments.downstream_models.transformer_enf import TransformerClassifier
from enf.bi_invariants import TranslationBI
from enf.utils import initialize_latents

jax.config.update("jax_default_matmul_precision", "highest")


def create_dataloaders(hdf5_path, endpoints_csv_path, batch_size=16, endpoint_name='LVEF', 
                      val_split=0.3, random_seed=42, num_workers=0, debug_limit=None, z_indices=None):
    """
    Create train and validation dataloaders for the autodecoding latent dataset.
    """
    # Convert tuple/list of -1 back to None for the dataset
    if z_indices == (-1,) or z_indices == [-1]:
        z_indices = None
    elif isinstance(z_indices, (tuple, str)):
        # Convert string representation of tuple to list
        if isinstance(z_indices, str):
            z_indices = eval(z_indices)
        z_indices = list(z_indices)
    
    try:
        full_dataset = LatentEndpointDatasetAutodecoding(
            hdf5_path, 
            endpoints_csv_path, 
            endpoint_name,
            debug_limit=debug_limit, 
            z_indices=z_indices
        )
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        print("WARNING: Dataset is empty!")
        return None, None
        
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create dataloaders
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    val_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    print(f"Created dataloaders with {train_size} training and {val_size} validation samples")
    return train_loader, val_loader


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.latent_path = '/projects/prjs1252/data_jesse_v2/latent_dataset_4d_autodecoding.h5'
    config.dataset.csv_path = '/projects/prjs1252/data_jesse/metadata/filtered_endpoints.csv'
    config.dataset.debug_limit = None
    config.dataset.z_indices = (-1,)  # Use tuple as default to indicate "all slices"

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_model = 1e-4

    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.hidden_size = 768
    config.model.num_classes = 2

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1
    config.train.num_epochs = 10
    config.train.validation_interval = 10

    # Experiment tracking
    config.exp_name = "lvef_prediction_autodecoding"
    config.run_name = "transformer_4d"
    
    return config


_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def main(_):
    config = _CONFIG.value
    logging.getLogger().setLevel(logging.INFO)

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online", name=config.run_name)

    # Create dataloaders
    train_dloader, val_dloader = create_dataloaders(
        hdf5_path=config.dataset.latent_path,
        endpoints_csv_path=config.dataset.csv_path,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit,
        z_indices=config.dataset.z_indices
    )
    
    logging.info(f"Train dataloader length: {len(train_dloader)}")
    logging.info(f"Val dataloader length: {len(val_dloader)}")

    # Get metadata from dataset
    with h5py.File(config.dataset.latent_path, 'r') as f:
        num_latents = f['metadata'].attrs['num_latents']
        latent_dim = f['metadata'].attrs['latent_dim']

    # Initialize model
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)

    # Create dummy latents for model initialization
    temp_z = initialize_latents(
        batch_size=1,
        num_latents=num_latents,
        latent_dim=latent_dim,
        data_dim=4,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.train.noise_scale,
    )

    # Define model
    model = TransformerClassifier(
        hidden_size=config.model.hidden_size,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=config.model.num_classes,
    )
    
    model_params = model.init(key, *temp_z)
    model_opt = optax.adam(learning_rate=config.optim.lr_model)
    model_opt_state = model_opt.init(model_params)
    
    # Compute context vector statistics for normalization
    c_list = []
    for _, z_tuples, _ in train_dloader:
        c_list.append(z_tuples[1])
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    
    @jax.jit
    def classifier_step(z, targets, model_params, model_opt_state):
        def cross_entropy_loss(params):
            p, c, g = z
            c = (c - c_mean) / c_std
            binary_targets = (targets >= 40.0).astype(jnp.int32)
            
            logits = model.apply(params, p, c, g)
            labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
            loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
            return loss

        loss, grads = jax.value_and_grad(cross_entropy_loss)(model_params)
        updates, model_opt_state = model_opt.update(grads, model_opt_state)
        model_params = optax.apply_updates(model_params, updates)

        return loss, model_params, model_opt_state

    # Training loop
    global_step = 0
    for epoch in range(config.train.num_epochs):
        epoch_losses = []
        
        for batch in tqdm(train_dloader, desc=f"Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            loss, model_params, model_opt_state = classifier_step(
                z, targets, model_params, model_opt_state
            )
            
            epoch_losses.append(loss)
            global_step += 1

            if global_step % config.train.validation_interval == 0:
                # Log training metrics
                avg_loss = jnp.mean(jnp.array(epoch_losses))
                logging.info(f"Step {global_step}: train-Loss = {avg_loss:.4f}")
                wandb.log({"train/loss": avg_loss}, step=global_step)
                epoch_losses = []
                
                # Validation loop
                val_losses = []
                val_accuracies = []
                misclassified_data = []
                
                for batch in tqdm(val_dloader, desc=f"Validation"):
                    patient_ids, z, targets = batch
                    p, c, g = z
                    c = (c - c_mean) / c_std
                    
                    logits = model.apply(model_params, p, c, g)
                    binary_targets = (targets >= 40.0).astype(jnp.int32)
                    
                    # Compute metrics
                    labels_onehot = jax.nn.one_hot(binary_targets, num_classes=2)
                    loss = -jnp.mean(jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1))
                    preds = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean((preds == binary_targets).astype(jnp.float32))
                    
                    # Track misclassified cases
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
                logging.info(f"Validation: Loss = {avg_val_loss:.4f}, Accuracy = {avg_val_accuracy:.4f}")
                
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/accuracy": avg_val_accuracy
                }, step=global_step)
                
                if misclassified_data:
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
                    wandb.log({"val/misclassified_patients": misclassified_table}, step=global_step)
    
    run.finish()


if __name__ == "__main__":
    app.run(main)
