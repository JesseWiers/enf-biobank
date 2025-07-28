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
from experiments.datasets.biobank_latent_dataset import create_dataloaders

from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerRegressor

jax.config.update("jax_default_matmul_precision", "highest")


def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.latent_path = '/projects/prjs1252/data_jesse/latent_dataset_4d.h5'
    config.dataset.csv_path = '/projects/prjs1252/data_jesse/metadata/filtered_endpoints.csv'
    config.dataset.z_indices = (3, 4, 5, 6)
    config.dataset.debug_limit = None

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_transformer = 1e-4
    
    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.hidden_size = 768

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train_cls = 10
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
    train_dloader, val_dloader, _ = create_dataloaders(
        hdf5_path=config.dataset.latent_path,
        endpoints_csv_path=config.dataset.csv_path,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        z_indices=list(config.dataset.z_indices),
        debug_limit=config.dataset.debug_limit
    )
    
    logging.info(f"Train dataloader length: {len(train_dloader)}")
    logging.info(f"Val dataloader length: {len(val_dloader)}")

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

    # Define the transformer model
    transformer = TransformerRegressor(
        hidden_size=config.model.hidden_size,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        output_dim=1,
    )
    transformer_params = transformer.init(key, *temp_z)

    # Define optimizer for the transformer model
    transformer_opt = optax.adam(learning_rate=config.optim.lr_transformer)
    transformer_opt_state = transformer_opt.init(transformer_params)
    
    
    # Compute mean and std of the context vectors for normalization for faster convergence
    c_list = []
    target_list = []
    for i, (_, z_tuples, targets) in enumerate(train_dloader):
        c_list.append(z_tuples[1])
        target_list.append(targets)
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    
    # Compute mean and std of targets for normalization
    target_mean = jnp.mean(jnp.concatenate(target_list, axis=0))
    target_std = jnp.std(jnp.concatenate(target_list, axis=0))
    logging.info(f"Target mean: {target_mean:.4f}, std: {target_std:.4f}")
    
    @jax.jit
    def regressor_step(z, targets, transformer_params, transformer_opt_state, key):
        
        def mse_loss(params):
            
            # Unpack the tuple
            p, c, g = z
            
            # Normalize the context vectors
            c = (c - c_mean) / c_std
            
            # Normalize targets
            targets_norm = (targets - target_mean) / target_std
            
            # Forward pass through transformer
            preds = transformer.apply(params, p, c, g)

            loss = jnp.mean((preds - targets_norm) ** 2)
                
            return loss

        # Get gradients
        loss, grads = jax.value_and_grad(mse_loss)(transformer_params)

        # Update transformer parameters
        updates, transformer_opt_state = transformer_opt.update(grads, transformer_opt_state)
        transformer_params = optax.apply_updates(transformer_params, updates)

        return loss, transformer_params, transformer_opt_state
    

    # Training loop
    for epoch in range(config.train.num_epochs_train_cls):
        epoch_losses = []
        
        # Add tqdm progress bar
        for batch in tqdm(train_dloader, desc=f"Epoch {epoch}"):
            patient_ids, z, targets = batch
            
            # Training step
            loss, transformer_params, transformer_opt_state = regressor_step(
                z, targets, transformer_params, transformer_opt_state, key
            )
            
            epoch_losses.append(loss)
        
        # Log metrics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        logging.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        wandb.log({"train-loss": avg_loss}, step=epoch)
        
    
        val_epoch_losses = []
        val_epoch_mse = []  # Track MSE in original scale
        # Validation loop
        for batch in tqdm(val_dloader, desc=f"Testing"):
            patient_ids, z, targets = batch
            
            # Unpack the tuple
            p, c, g = z
            
            # Normalize the context vectors
            c = (c - c_mean) / c_std
            
            # Forward pass through transformer (predictions are in normalized space)
            preds_norm = transformer.apply(transformer_params, p, c, g)
            
            # Calculate MSE in normalized space
            loss = jnp.mean((preds_norm - (targets - target_mean) / target_std) ** 2)
            val_epoch_losses.append(loss)
            
            # Denormalize predictions for interpretable MSE
            preds = preds_norm * target_std + target_mean
            mse = jnp.mean((preds - targets) ** 2)
            val_epoch_mse.append(mse)
            
        # Log metrics
        avg_val_loss = jnp.mean(jnp.array(val_epoch_losses))
        avg_val_mse = jnp.mean(jnp.array(val_epoch_mse))
        logging.info(f"Epoch {epoch}: Normalized Loss = {avg_val_loss:.4f}, MSE = {avg_val_mse:.4f}")
        wandb.log({
            "val-loss-normalized": avg_val_loss,
            "val-mse": avg_val_mse
        }, step=epoch)
            
    run.finish()


if __name__ == "__main__":
    app.run(main)
