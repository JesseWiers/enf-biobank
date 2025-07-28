import ml_collections
from ml_collections import config_flags
from absl import app, flags

import jax
import jax.numpy as jnp
import optax
import logging
import matplotlib.pyplot as plt
import wandb
import os
import pickle
import time

from flax.training import checkpoints
import flax
flax.training.checkpoints.CHECKPOINT_GDA = False

# We'll need to create these
import torch  # For DataLoader
import haiku as hk  # For VAE
from jax_vae.vae_continuous import (
    VariationalAutoEncoder, 
    mean_squared_error, 
    kl_gaussian
)  # Import loss functions too
from experiments.datasets.vae_dataset import CardiacPatchDataset
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from jax_vae.vae_version2 import BetterVAE


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.save_checkpoints = False
    config.checkpoint_path = ""
    config.exp_name = "vae_training"
    config.run_name = "better_vae_progressive_patch_16_lr_1e-4_kl_1"

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.nifti_dir = "/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped"
    config.dataset.patch_size = 16
    config.dataset.target_size = 80
    config.dataset.num_workers = 4

    # Training config
    config.training = ml_collections.ConfigDict()  # Changed from train to training
    config.training.batch_size = 32
    config.training.learning_rate = 1e-4
    config.training.num_steps = 10000
    config.training.log_every = 100

    # VAE config
    config.vae = ml_collections.ConfigDict()
    config.vae.version = "v2"
    config.vae.hidden_dims = "32,64,128,256"
    config.vae.hidden_size1 = 512  # Add these two lines
    config.vae.hidden_size2 = 256  # for v1 compatibility
    config.vae.latent_size = 64
    config.vae.output_shape = (16, 16, 1)
    config.vae.kl_weight = 1.0
    config.vae.batch_norm_decay = 0.9

    return config

# Define the config flag BEFORE main
_CONFIG = config_flags.DEFINE_config_dict('config', get_config())

def plot_vae_results(patches, reconstructed, patch_size, n_examples=5):
    """Plot original and reconstructed patches side by side"""
    fig, axes = plt.subplots(2, n_examples, figsize=(2*n_examples, 4))
    
    for i in range(n_examples):
        # Original patch
        axes[0, i].imshow(patches[i].reshape(patch_size, patch_size), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
            
        # Reconstructed patch
        axes[1, i].imshow(reconstructed[i].reshape(patch_size, patch_size), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    return fig

def compute_psnr(original, reconstruction, max_val=1.0):
    """Compute PSNR between original and reconstruction"""
    mse = jnp.mean((original - reconstruction) ** 2)
    psnr = 20 * jnp.log10(max_val / jnp.sqrt(mse))
    return psnr

def train_step(
    model: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    batch: jnp.ndarray,
    rng: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    config: ml_collections.ConfigDict,  # Add config parameter
) -> Tuple[hk.Params, hk.State, optax.OptState, Dict[str, float]]:
    """Single training step."""
    
    def loss_fn(params: hk.Params, state: hk.State, batch: jnp.ndarray, rng: jnp.ndarray):
        (output, new_state) = model.apply(params, state, rng, batch, is_training=True)
        recon_loss = jnp.mean((output.output - batch) ** 2)
        kl_loss = -0.5 * jnp.mean(1 + jnp.log(output.stddev ** 2) - output.mean ** 2 - output.stddev ** 2)
        total_loss = recon_loss + config.vae.kl_weight * kl_loss
        return total_loss, (new_state, recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, recon_loss, kl_loss)), grads = grad_fn(params, state, batch, rng)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    metrics = {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }
    
    return params, new_state, opt_state, metrics

def train_step_v1(
    model: hk.Transformed,  # Changed from Transform to Transformed
    params: hk.Params,
    batch: jnp.ndarray,
    rng: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    config: ml_collections.ConfigDict,
) -> Tuple[hk.Params, optax.OptState, Dict[str, float]]:
    """Single training step for v1 VAE."""
    
    def loss_fn(params: hk.Params, batch: jnp.ndarray, rng: jnp.ndarray):
        output = model.apply(params, rng, batch)
        recon_loss = jnp.mean((output.output - batch) ** 2)
        kl_loss = -0.5 * jnp.mean(1 + jnp.log(output.stddev ** 2) - output.mean ** 2 - output.stddev ** 2)
        total_loss = recon_loss + config.vae.kl_weight * kl_loss
        return total_loss, (recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon_loss, kl_loss)), grads = grad_fn(params, batch, rng)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    metrics = {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }
    
    return params, opt_state, metrics

def main(_):
    config = _CONFIG.value
    
    # Initialize wandb
    run = wandb.init(
        project=config.exp_name,
        config=config.to_dict(),
        mode="online" if not config.debug else "dryrun",
        name=config.run_name
    )
    
    # Log JAX device info
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Set random seed
    rng = jax.random.PRNGKey(config.seed)
    rng_seq = hk.PRNGSequence(rng)
    
    # Create datasets
    train_dataset = CardiacPatchDataset(
        nifti_dir=config.dataset.nifti_dir,
        patch_size=config.dataset.patch_size,
        target_size=config.dataset.target_size,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    train_iter = iter(train_loader)
    
    # Log dataset sizes
    logging.info("Dataset sizes:")
    logging.info(f"  Total:      {len(train_dataset)}")
    logging.info(f"  Training:   {len(train_dataset) - 100}")
    logging.info(f"  Validation: 100")
    
    # Initialize model
    dummy_batch = jnp.zeros((1, *config.vae.output_shape))
    if config.vae.version == "v1":
        model = hk.transform(lambda x: VariationalAutoEncoder(
            hidden_size1=config.vae.hidden_size1,
            hidden_size2=config.vae.hidden_size2,
            latent_size=config.vae.latent_size,
            output_shape=config.vae.output_shape,
        )(x))
        params = model.init(next(rng_seq), dummy_batch)
        state = None
    else:  # v2
        model = hk.transform_with_state(lambda x, is_training: BetterVAE(
            latent_size=config.vae.latent_size,
            hidden_dims=[int(x) for x in config.vae.hidden_dims.split(',')],
            output_shape=config.vae.output_shape,
        )(x, is_training=is_training))
        params, state = model.init(next(rng_seq), dummy_batch, is_training=True)
    
    # Initialize optimizer
    optimizer = optax.adam(config.training.learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    num_steps = config.training.num_steps # Use num_epochs * steps per epoch
    log_every = config.training.log_every
    
    # Initialize metrics
    running_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    for step in range(num_steps):
        batch = next(train_iter)['patches'].reshape(-1, *config.vae.output_shape)
        batch = jnp.array(batch)
        
        if config.vae.version == "v1":
            # V1 training step
            params, opt_state, metrics = train_step_v1(
                model, params, batch, next(rng_seq), optimizer, opt_state, config
            )
            new_state = None
        else:
            # V2 training step with state
            params, state, opt_state, metrics = train_step(
                model, params, state, batch, next(rng_seq), optimizer, opt_state, config
            )
        
        # Update running averages
        running_loss = 0.9 * running_loss + 0.1 * metrics["loss"]
        running_recon_loss = 0.9 * running_recon_loss + 0.1 * metrics["recon_loss"]
        running_kl_loss = 0.9 * running_kl_loss + 0.1 * metrics["kl_loss"]
        
        # Log metrics
        if (step + 1) % log_every == 0:
            wandb.log({
                "loss": running_loss,
                "recon_loss": running_recon_loss,
                "kl_loss": running_kl_loss,
                "step": step,
            })
            
            # Log reconstructions
            if config.vae.version == "v1":
                output = model.apply(params, next(rng_seq), batch)
            else:
                output, _ = model.apply(params, state, next(rng_seq), batch, is_training=False)
            
            fig, ax = plt.subplots(2, 8, figsize=(20, 5))
            for i in range(8):
                ax[0, i].imshow(batch[i, ..., 0], cmap='gray')
                ax[0, i].axis('off')
                ax[1, i].imshow(output.output[i, ..., 0], cmap='gray')
                ax[1, i].axis('off')
            wandb.log({"reconstructions": wandb.Image(fig)})
            plt.close()
            
            logging.info(f"Step {step + 1}/{num_steps}")
            logging.info(f"Loss: {running_loss:.4f}")
            logging.info(f"Recon Loss: {running_recon_loss:.4f}")
            logging.info(f"KL Loss: {running_kl_loss:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    app.run(main)
