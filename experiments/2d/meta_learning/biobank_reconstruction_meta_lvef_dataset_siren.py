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

# Custom imports
from experiments.datasets import get_dataloaders
from enf.utils import create_coordinate_grid  # Keep this import
from experiments.downstream_models.siren_jax.siren.network import create_modulated_mlp
from experiments.downstream_models.siren_jax.siren.layer import ModulatedDense

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
    config.siren.num_layers = 5
    config.siren.omega = 30.0  # frequency multiplier
    config.siren.w0_increment = 0.0  # increment omega per layer if desired
    config.siren.latent_modulation_dim = 512  # Add this
    config.siren.modulate_shift = True        # Add this
    config.siren.modulate_scale = False       # Add this

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_siren = 5e-4
    config.optim.inner_lr = 1e-2  # Single learning rate
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

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

def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('biobank_lvef', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed)
    sample_img, _, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:]  

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)
    
    # Initialize ModulatedSIREN
    net_params, net_apply = create_modulated_mlp(
        input_dim=2,
        latent_dim=config.siren.latent_modulation_dim,
        num_channels=[config.siren.hidden_size] * (config.siren.num_layers - 1),
        output_dim=1,
        omega=config.siren.omega,
        modulate_shift=config.siren.modulate_shift,
        modulate_scale=config.siren.modulate_scale
    )

    # Define optimizer for SIREN
    siren_opt = optax.adam(learning_rate=config.optim.lr_siren)
    siren_opt_state = siren_opt.init(net_params)


    @jax.jit
    def recon_inner_loop(net_params, coords, img, key):
        # Reshape coordinates and image if needed
        batch_size = coords.shape[0]
        coords = coords.reshape(batch_size, -1, 2)
        img = img.reshape(batch_size, -1, 1)

        # Initialize modulations
        modulations = jax.random.normal(
            key, 
            (batch_size, config.siren.latent_modulation_dim)
        ) * 0.1  # Small initialization

        def mse_loss(modulations):
            out = net_apply(net_params, coords, modulations)  # Note the additional modulations argument
            return jnp.mean((out - img) ** 2)

        def psnr(modulations):
            out = net_apply(net_params, coords, modulations)  # Note the additional modulations argument
            mse = jnp.mean((img - out) ** 2, axis=(1, 2))
            return 20 * jnp.log10(1.0 / jnp.sqrt(mse))

        def inner_step(modulations, _):
            loss, grads = jax.value_and_grad(mse_loss)(modulations)
            # Update modulations instead of network parameters
            modulations = modulations - config.optim.inner_lr * grads
            return modulations, loss

        # Perform inner loop optimization
        modulations, losses = jax.lax.scan(
            inner_step,
            modulations,
            None,
            length=config.optim.inner_steps
        )

        if config.optim.first_order_maml:
            modulations = jax.lax.stop_gradient(modulations)

        final_loss = mse_loss(modulations)
        psnr_value = psnr(modulations)
        
        return final_loss, (modulations, psnr_value)

    @jax.jit
    def recon_outer_step(coords, img, net_params, siren_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, (modulations, psnr_value)), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(net_params, coords, img, key)

        # Update the network parameters
        updates, siren_opt_state = siren_opt.update(grads, siren_opt_state)
        net_params = optax.apply_updates(net_params, updates)

        return (loss, modulations, psnr_value), net_params, siren_opt_state, subkey
    
    
    def evaluate_test_set(net_params, test_dloader, key):
        """Evaluate model on the entire test set."""
        psnrs = []
        mses = []
        
        # Randomly select which batch to return for visualization
        key, subkey = jax.random.split(key)
        batch_idx_to_return = jax.random.randint(subkey, (), 0, len(test_dloader))
        
        vis_batch_original = None
        vis_batch_recon = None
        
        for i, (img, _, _) in enumerate(test_dloader):
            # Reshape input
            batch_size = img.shape[0]
            y = img.reshape(batch_size, -1, 1)
            coords = x.reshape(batch_size, -1, 2)
            
            key, subkey = jax.random.split(key)
            mse, (modulations, psnr_value) = recon_inner_loop(net_params, coords, y, key)
            
            psnrs.append(psnr_value)
            mses.append(mse)
            
            # Store reconstruction for randomly selected batch
            if i == batch_idx_to_return:
                vis_batch_original = img
                vis_batch_recon = net_apply(net_params, coords, modulations).reshape(img.shape)  # Use net_apply with modulations
        
        # Calculate averages
        avg_psnr = jnp.mean(jnp.array(psnrs))
        avg_mse = jnp.mean(jnp.array(mses))
        
        # Return metrics and visualization batch
        return avg_psnr, avg_mse, vis_batch_original, vis_batch_recon, 0.0  # 0.0 for context_std since SIREN doesn't use it
    
    if os.path.exists(config.checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {config.checkpoint_path}\033[0m")
        net_params, siren_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(config.checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
    
 
    # Pretraining loop for fitting the SIREN backbone
    best_psnr = float('-inf')
    glob_step = 0
    for epoch in range(config.train.num_epochs_train):
        epoch_loss = []
        epoch_psnr = []
        
        # Training loop
        for i, (img, _, _) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Perform outer loop optimization
            (loss, modulations, psnr_value), net_params, siren_opt_state, key = recon_outer_step(
                x, y, net_params, siren_opt_state, key)

            epoch_loss.append(loss)
            epoch_psnr.append(psnr_value)
            glob_step += 1

        # Evaluation at the end of each epoch
        test_psnr, test_mse, img, img_r, context_std = evaluate_test_set(net_params, test_dloader, key)
        
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
