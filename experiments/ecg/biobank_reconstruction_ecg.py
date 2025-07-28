import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt

import wandb

import os
import pickle

import numpy as np

from experiments.ecg.dataloader import ECGLeadDataset, numpy_collate, to_numpy

import torch
import torchvision
from torch.utils.data import DataLoader

from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

jax.config.update("jax_default_matmul_precision", "highest")


def save_checkpoint(checkpoint_dir, model_params, optimizer_state, epoch, global_step, best_psnr):
    """
    Saves the model parameters, optimizer state, and training metadata.
    
    Args:
        checkpoint_dir (str): Directory where the checkpoint will be saved.
        model_params (PyTree): The parameters of the model to be saved.
        optimizer_state (PyTree): The state of the optimizer.
        epoch (int): Current epoch number.
        global_step (int): Current training step.
        best_psnr (float): The best PSNR value encountered so far.
    """
    
    # Ensure the checkpoint directory is an absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Construct checkpoint dictionary
    checkpoint_data = {
        "model_params": model_params,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pkl")
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
        
    print(f"Checkpoint saved at step {global_step} in {checkpoint_dir}")
    
    
def load_checkpoint(checkpoint_path):
    """
    Loads a checkpoint from a .pkl file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return (
        checkpoint_data["model_params"],
        checkpoint_data["optimizer_state"],
        checkpoint_data["epoch"],
        checkpoint_data["global_step"],
        checkpoint_data["best_psnr"],
    )

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "ecg_reconstruction"
    config.exp_name = "ecg_test"
    config.save_checkpoints = True
    config.checkpoint_path = ""

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 1  # ECG is 1D
    config.recon_enf.num_out = 1
    config.recon_enf.freq_mult = (30.0, 60.0)  # Adjust for ECG frequencies
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True
    config.recon_enf.num_latents = 64  # Maybe increase for longer sequences
    config.recon_enf.latent_dim = 32
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.lead_idx = 0  # Which ECG lead to use
    config.dataset.base_path = '/home/jwiers/deeprisk/new_codebase/enf-min-jax-version-1/data/biobank_ecg'

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 32  # Can be larger for 1D data
    config.train.noise_scale = 1e-1
    config.train.num_epochs = 100
    config.train.validation_interval = 10

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (2., 30., 0.)  # (pose, context, window)
    config.optim.inner_steps = 3

    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
):
    """Plot original and reconstructed biobank images side by side.
    
    Args:
        original: Original images with shape (B, H, W, 1) or (1, H, W, 1)
        reconstruction: Reconstructed images with shape (B, H, W, 1) or (1, H, W, 1)
        poses: Optional poses to plot on the image
    """
    # Squeeze out batch and channel dimensions to get 2D arrays (H, W)
    original = jnp.squeeze(original)
    reconstruction = jnp.squeeze(reconstruction)
    
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    fig.suptitle('Original vs Reconstruction')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')

    # Plot reconstructed
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction')

    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_ecg_reconstruction(original, reconstruction, loss, psnr, sample_rate=500):
    """Plot original and reconstructed ECG signals.
    
    Args:
        original: Original ECG with shape (batch, time, 1)
        reconstruction: Reconstructed ECG with shape (batch, time, 1)
        loss: MSE loss value
        psnr: PSNR value
        sample_rate: Sampling rate in Hz
    """
    batch_size = original.shape[0]
    time = np.linspace(0, original.shape[1]/sample_rate, original.shape[1])
    
    fig, axes = plt.subplots(batch_size, 1, figsize=(15, 3*batch_size))
    if batch_size == 1:
        axes = [axes]
        
    for i in range(batch_size):
        axes[i].plot(time, original[i, :, 0], 'b-', label='Original', alpha=0.7)
        axes[i].plot(time, reconstruction[i, :, 0], 'r--', label='Reconstructed', alpha=0.7)
        axes[i].set_title(f'ECG Lead Reconstruction (Loss: {loss:.4f}, PSNR: {psnr:.2f})')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    return fig

def main(_):
    config = _CONFIG.value
    
    # Initialize wandb
    run = wandb.init(
        project=config.exp_name,
        config=config.to_dict(),
        mode="online" if not config.debug else "dryrun",
        name=config.run_name
    )

    # Create dataloaders
    transforms = torchvision.transforms.Compose([to_numpy])
    train_dset = ECGLeadDataset(split='train', transforms=transforms, lead_idx=config.dataset.lead_idx)
    val_dset = ECGLeadDataset(split='val', transforms=transforms, lead_idx=config.dataset.lead_idx)
    
    train_dloader = DataLoader(
        train_dset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
        num_workers=config.dataset.num_workers
    )
    
    val_dloader = DataLoader(
        val_dset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        num_workers=config.dataset.num_workers
    )

    # Create coordinate grid
    sample_inp, _ = next(iter(train_dloader))
    inp_shape = sample_inp.shape[1:]
    x = create_coordinate_grid(
        img_shape=inp_shape,
        batch_size=config.train.batch_size,
        num_in=config.recon_enf.num_in
    )

    # Define the reconstruction and segmentation models
    recon_enf = EquivariantNeuralField(
        num_hidden=config.recon_enf.num_hidden,
        att_dim=config.recon_enf.att_dim,
        num_heads=config.recon_enf.num_heads,
        num_out=config.recon_enf.num_out,
        emb_freq=config.recon_enf.freq_mult,
        nearest_k=config.recon_enf.k_nearest,
        bi_invariant=TranslationBI(),
    )
        
    # Create dummy latents for model init
    key = jax.random.PRNGKey(55)
    temp_z = initialize_latents(
        batch_size=config.train.batch_size,  # Only need one example for initialization
        num_latents=config.recon_enf.num_latents,
        latent_dim=config.recon_enf.latent_dim,
        data_dim=config.recon_enf.num_in,
        bi_invariant_cls=TranslationBI,
        key=key,
        noise_scale=config.train.noise_scale,
    )


    enf_params = recon_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    enf_opt_state = enf_opt.init(enf_params)
    
    @jax.jit
    def recon_inner_loop(enf_params, coords, img, key):

        z = initialize_latents(
            batch_size=config.train.batch_size,
            num_latents=config.recon_enf.num_latents,
            latent_dim=config.recon_enf.latent_dim,
            data_dim=config.recon_enf.num_in,
            bi_invariant_cls=TranslationBI,
            key=key,
            noise_scale=config.train.noise_scale,
        )

        def mse_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1,2)))
        
        def psnr(z):
            out = recon_enf.apply(enf_params, coords, *z)

            mse = jnp.mean((img - out) ** 2, axis=(1, 2))
            
            max_pixel_value = 1.0 
            psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
            
            return jnp.mean(psnr)
            
        def update_latents(z, _):
            _, grads = jax.value_and_grad(mse_loss)(z)
            
            # Gradient descent update, updating each leaf of z
            z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, grads, config.optim.inner_lr)
            return z, None
        
        # Perform inner loop optimization. jax.lax.scan is used to perform the optimization over multiple steps (optim.inner_steps)
        z, _ = jax.lax.scan(update_latents, z, None, length=config.optim.inner_steps)

        # Stop gradient if first order MAML (Model Agnostic Meta Learning)
        if config.optim.first_order_maml:
            z = jax.lax.stop_gradient(z)
            
        psnr_value = psnr(z)

        # Stop gradient for psnr_value
        psnr_value = jax.lax.stop_gradient(psnr_value)

        # Return loss, z, and psnr_value. Integration happens with respect to the first output (mse_loss(z))
        # auxiliary outputs are returned as a tuple
        return mse_loss(z), (z, psnr_value)


    @jax.jit
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        
        # compute gradients with respect to enf parameters. aux=true allows to return outputs not related to gradients
        (loss, (z, psnr_value)), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key. You can now use psnr_value here if needed
        return (loss, z, psnr_value), enf_params, enf_opt_state, subkey
    
    
    # Training loop
    best_psnr = float('-inf')
    glob_step = 0
    for epoch in range(config.train.num_epochs):
        epoch_loss = []
        epoch_psnr = []
        
        for batch in tqdm(train_dloader, desc=f"Epoch {epoch}"):
            ecg, _ = batch
            (loss, z, psnr_value), enf_params, enf_opt_state, key = recon_outer_step(
                x, ecg, enf_params, enf_opt_state, key)
            
            epoch_loss.append(loss)
            epoch_psnr.append(psnr_value)
            glob_step += 1

        # Validation
        if epoch % config.train.validation_interval == 0:
            val_losses = []
            val_psnrs = []
            example_batch = None
            
            for batch in tqdm(val_dloader, desc="Validation"):
                ecg, _ = batch
                loss, (z, psnr) = recon_inner_loop(enf_params, x, ecg, key)
                ecg_r = recon_enf.apply(enf_params, x, *z)
                
                val_losses.append(loss)
                val_psnrs.append(psnr)
                
                if example_batch is None:
                    example_batch = (ecg, ecg_r, loss, psnr)

            # Log metrics
            avg_loss = np.mean(epoch_loss)
            avg_psnr = np.mean(epoch_psnr)
            val_loss = np.mean(val_losses)
            val_psnr = np.mean(val_psnrs)
            
            # Create visualization
            fig = plot_ecg_reconstruction(*example_batch)
            
            wandb.log({
                "train/loss": avg_loss,
                "train/psnr": avg_psnr,
                "val/loss": val_loss,
                "val/psnr": val_psnr,
                "reconstruction": fig,
                "epoch": epoch
            })
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                if config.save_checkpoints:
                    save_checkpoint(
                        f"model_checkpoints/{config.exp_name}/{config.run_name}",
                        enf_params, enf_opt_state, epoch, glob_step, best_psnr
                    )
            
            plt.close('all')
    
    run.finish()


if __name__ == "__main__":
    app.run(main)
