import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging

import matplotlib.pyplot as plt

import wandb

import os
import pickle

from flax.training import checkpoints
import flax
flax.training.checkpoints.CHECKPOINT_GDA = False

from functools import partial

# Custom imports
from experiments.datasets import get_dataloaders
from enf.utils import create_coordinate_grid  # Keep this import
from experiments.downstream_models.siren_jax.siren.network import ModulatedSIREN
from experiments.downstream_models.siren_jax.siren.layer import ModulatedDense

from tqdm import tqdm

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
    config.run_name = "biobank_reconstruction"
    config.exp_name = "test"
    config.save_checkpoints = False
    config.checkpoint_path = ""

    # SIREN config
    config.siren = ml_collections.ConfigDict()
    config.siren.hidden_size = 256
    config.siren.latent_modulation_dim = 2048  # Add this
    config.siren.num_layers = 15 
    
    config.siren.omega = 1.0  # frequency multiplier (γ in paper) -> Had 30.0 myself before
    config.siren.w0_increment = 0.0  # increment omega per layer if desired
    config.siren.modulate_shift = True        # Add this
    config.siren.modulate_scale = False       # Add this

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_siren = 5e-4  # Network learning rate
    config.optim.latent_lr = 1e-2  # Latent vectors learning rate
    
    # Remove meta-learning specific configs
    # config.optim.inner_lr = 1e-2  # Not needed anymore
    # config.optim.inner_steps = 3  # Not needed anymore
    # config.optim.first_order_maml = False  # Not needed anymore

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.num_epochs_train = 10
    config.train.log_interval = 50

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

@partial(jax.jit, static_argnames=('siren', 'siren_opt', 'latent_lr', 'training'))
def train_step(net_params, latents, coords, img, siren, siren_opt, siren_opt_state, latent_lr, training=True):
    def loss_fn(params_and_latents):
        net_params, latents = params_and_latents
        # Replace net_apply with siren.__call__
        out = siren(net_params, coords, latents)
        return jnp.mean((out - img) ** 2)

    def psnr(params_and_latents):
        net_params, latents = params_and_latents
        out = siren(net_params, coords, latents)
        mse = jnp.mean((img - out) ** 2, axis=(1, 2))
        return 20 * jnp.log10(1.0 / jnp.sqrt(mse))

    loss, grads = jax.value_and_grad(loss_fn)((net_params, latents))
    net_grads, latent_grads = grads

    if training:
        updates, new_siren_opt_state = siren_opt.update(net_grads, siren_opt_state)
        net_params = optax.apply_updates(net_params, updates)
        latents = latents - latent_lr * latent_grads  # Use latent_lr directly
    else:
        new_siren_opt_state = siren_opt_state

    psnr_value = psnr((net_params, latents))
    return loss, (net_params, latents, psnr_value), new_siren_opt_state


def evaluate_test_set(net_params, test_dloader, siren, latents, x, siren_opt, siren_opt_state, config, key):
    psnrs = []
    mses = []
    
    key, subkey = jax.random.split(key)
    batch_idx_to_return = jax.random.randint(subkey, (), 0, len(test_dloader))
    
    vis_batch_original = None
    vis_batch_recon = None
    
    for i, (img, seg, indices) in enumerate(test_dloader):
        batch_size = img.shape[0]
        y = img.reshape(batch_size, -1, 1)
        # Adjust coordinate grid for current batch size if needed
        if batch_size != x.shape[0]:
            coords = create_coordinate_grid(batch_size=batch_size, img_shape=img.shape[1:3])
        else:
            coords = x
        coords = coords.reshape(batch_size, -1, 2)
        batch_latents = latents[indices]
        
        key, subkey = jax.random.split(key)
        mse, (_, _, psnr_value), _ = train_step(
            net_params, batch_latents, coords, y, siren, siren_opt, 
            siren_opt_state, config.optim.latent_lr, training=False)
        
        psnrs.append(psnr_value)
        mses.append(mse)
        
        if i == batch_idx_to_return:
            vis_batch_original = img
            vis_batch_recon = siren(net_params, coords, batch_latents).reshape(img.shape)
    
    avg_psnr = jnp.mean(jnp.array(psnrs))
    avg_mse = jnp.mean(jnp.array(mses))
    
    return avg_psnr, avg_mse, vis_batch_original, vis_batch_recon, 0.0


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders(
        'biobank_lvef', 
        config.train.batch_size, 
        config.dataset.num_workers, 
        num_train=config.dataset.num_patients_train, 
        num_test=config.dataset.num_patients_test, 
        seed=config.seed,
    )
    sample_img, sample_seg, sample_idx = next(iter(train_dloader))
    img_shape = sample_img.shape[1:]

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)
    
    # Initialize ModulatedSIREN
    siren = ModulatedSIREN(config)
    key, subkey = jax.random.split(key)
    net_params = siren.init(
        subkey, 
        input_shape=(2,),  # 2D coordinates
        latent_dim=config.siren.latent_modulation_dim
    )

    # Define optimizer for SIREN
    siren_opt = optax.adam(learning_rate=config.optim.lr_siren)
    siren_opt_state = siren_opt.init(net_params)


    if os.path.exists(config.checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {config.checkpoint_path}\033[0m")
        net_params, siren_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(config.checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
    
 
    # Pretraining loop for fitting the SIREN backbone
    # Initialize latent vectors for all training images
    num_train_images = len(train_dloader.dataset)
    key, subkey = jax.random.split(key)
    latents = jax.random.normal(
        subkey, 
        (num_train_images, config.siren.latent_modulation_dim)
    ) * 0.1

    best_psnr = float('-inf')
    glob_step = 0

    # Add tqdm for epochs
    for epoch in tqdm(range(config.train.num_epochs_train), desc="Training epochs"):
        epoch_loss = []
        epoch_psnr = []
        
        # Add tqdm for batches within each epoch
        for i, (img, seg, indices) in enumerate(tqdm(train_dloader, desc=f"Epoch {epoch}", leave=False)):
            batch_size = img.shape[0]
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Match coordinate grid handling with evaluation
            if batch_size != x.shape[0]:
                coords = create_coordinate_grid(batch_size=batch_size, img_shape=img.shape[1:3])
            else:
                coords = x
            coords = coords.reshape(batch_size, -1, 2)
            
            batch_latents = latents[indices]

            loss, (net_params, updated_latents, psnr_value), siren_opt_state = train_step(
                net_params, batch_latents, coords, y, siren, siren_opt, 
                siren_opt_state, config.optim.latent_lr, training=True)

            latents = latents.at[indices].set(updated_latents)
            epoch_loss.append(loss)
            epoch_psnr.append(psnr_value)
            glob_step += 1

        # Evaluation
        test_psnr, test_mse, img, img_r, context_std = evaluate_test_set(
            net_params, test_dloader, siren, latents, x, siren_opt, 
            siren_opt_state, config, key)
        
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            if config.save_checkpoints:
                save_checkpoint(f"model_checkpoints/{config.exp_name}/{config.run_name}", 
                              net_params, siren_opt_state, epoch, glob_step, best_psnr)
        
        # Visualization and logging
        random_idx = jax.random.randint(key, (1,), 0, config.train.batch_size)
        fig = plot_biobank_comparison(img[random_idx], img_r[random_idx])
        
        # Calculate average loss for the epoch
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        
        wandb.log({
            "recon-mse": avg_epoch_loss, 
            "test-mse": test_mse, 
            "test-psnr": test_psnr,
            "context-std": context_std,
            "reconstruction": fig,
            "epoch": epoch
        }, step=glob_step)
        plt.close('all')
        
        logging.info(f"RECON ep {epoch} || mse: {avg_epoch_loss} || test-mse: {test_mse} || test-psnr: {test_psnr} || context-std: {context_std}")

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
