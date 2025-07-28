import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import wandb
import time
import os
import pickle
# Custom imports
from experiments.datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

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
        
    logging.info(f"Checkpoint saved at step {global_step} in {checkpoint_dir}")
    
    
def load_checkpoint(checkpoint_path):
    """
    Loads a checkpoint from a .pkl file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    logging.info(f"psnr of checkpoint: {checkpoint_data['best_psnr']}")
    
    return (
        checkpoint_data["model_params"],
        checkpoint_data["optimizer_state"],
        checkpoint_data["epoch"],
        checkpoint_data["global_step"],
        checkpoint_data["best_psnr"],
    )
    
    
def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "biobank_reconstruction"
    config.exp_name = "test"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 2  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 512
    config.recon_enf.latent_dim = 64
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 50 
    config.dataset.num_patients_test = 10

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4 
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.) # NOTE: Try 1e-3 
    config.optim.first_order_maml = False

    # Training config
    config.eval = ml_collections.ConfigDict()
    config.eval.batch_size = 16
    config.eval.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.eval.num_epochs = 10
    config.eval.log_interval = 200
    config.eval.checkpoint_path = ""
    logging.getLogger().setLevel(logging.INFO)

    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())

def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
):
    """Plot original and reconstructed CIFAR images side by side.
    
    Args:
        original: Original images with shape (H, W, 3)
        reconstruction: Reconstructed images with shape (H, W, 3)
        poses: Optional poses to plot on the image
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    fig.suptitle('Original (top) vs Reconstruction (bottom)')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')

    # Plot reconstructed
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction')

    # Plot poses
    # if poses is not None:
    #     # Map to 0-W range
    #     poses = (poses + 1) * original.shape[0] / 2
    #     axes[2].imshow(reconstruction, cmap='gray')
    #     axes[2].scatter(poses[:, 0], poses[:, 1], c='r', s=2)
    #     axes[2].set_title('Poses')

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
    _, test_dloader = get_dataloaders('2d_biobank_v2', config.eval.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed, shuffle_train=False)
    sample_img, _ = next(iter(test_dloader))
    img_shape = sample_img.shape[1:] 

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.eval.batch_size, img_shape=img_shape)

    # Define the reconstruction model
    recon_enf = EquivariantNeuralField(
        num_hidden=config.recon_enf.num_hidden,
        att_dim=config.recon_enf.att_dim,
        num_heads=config.recon_enf.num_heads,
        num_out=config.recon_enf.num_out,
        emb_freq=config.recon_enf.freq_mult,
        nearest_k=config.recon_enf.k_nearest,
        bi_invariant=TranslationBI(),
        gaussian_window=config.recon_enf.gaussian_window,
    )

    # Create dummy latents for model init
    key, subkey = jax.random.split(key)
    temp_z = initialize_latents(
        batch_size=1,  # Only need one example for initialization
        num_latents=config.recon_enf.num_latents,
        latent_dim=config.recon_enf.latent_dim,
        data_dim=config.recon_enf.num_in,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.eval.noise_scale,
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)


    # Loading checkpoint
    if not os.path.exists(config.eval.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.eval.checkpoint_path}")
        
    logging.info(f"\033[93mLoading trained model from checkpoint: {config.eval.checkpoint_path}\033[0m")
    recon_enf_params, _, _, _, _ = load_checkpoint(config.eval.checkpoint_path)
    
    # Initialize latents for all test samples
    num_test_samples = len(test_dloader) * config.eval.batch_size
    logging.info(f"Initializing latents for {num_test_samples} test samples")
    
    key, subkey = jax.random.split(key)
    z_test_dataset = initialize_latents(
        batch_size=num_test_samples,
        num_latents=config.recon_enf.num_latents,
        latent_dim=config.recon_enf.latent_dim,
        data_dim=config.recon_enf.num_in,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.eval.noise_scale,
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )
       
    # Create optimization step for latents only
    @jax.jit
    def optimize_latent_step(coords, img, z, enf_params, key):
        def mse_loss(z):  
            out = recon_enf.apply(enf_params, coords, *z)  
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)

        key, subkey = jax.random.split(key)
        loss, z_grads = jax.value_and_grad(mse_loss)(z)
        
        # Update latents only, using the inner_lr from config
        z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, z_grads, config.optim.inner_lr)
        
        return (loss, z), subkey
    
    @jax.jit
    def evaluate_batch(enf_params, coords, img, z):
        # Compute reconstruction
        img_r = recon_enf.apply(enf_params, coords, *z).reshape(img.shape)
        
        # Compute PSNR
        mse = jnp.mean((img - img_r) ** 2)
        max_pixel_value = 1.0 
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
        return psnr
    
    best_psnr = float('-inf')
    
    for epoch in range(config.eval.num_epochs):
        epoch_loss = []
        
        # Time epoch
        start_time = time.time()
        
        # Update latents for each batch
        for i, (img, _) in enumerate(tqdm(test_dloader, desc=f"Eval Epoch {epoch}")):
            
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Extract latents corresponding to batch
            z = jax.tree.map(lambda x: x[i*config.eval.batch_size:(i+1)*config.eval.batch_size], z_test_dataset)
            
            # Update latents for this batch
            (loss, z), key = optimize_latent_step(x, y, z, recon_enf_params, key)
            
            # Update dataset with new latents
            z_test_dataset = jax.tree.map(
                lambda full, partial: full.at[i*config.eval.batch_size:(i+1)*config.eval.batch_size].set(partial), 
                z_test_dataset, z
            )
            
            epoch_loss.append(loss)
            
        end_time = time.time()
        logging.info(f"Eval Epoch {epoch} took {end_time - start_time} seconds")
        
        # Evaluate on test set
        psnrs_evaluation = []
        
        for i, (img, _) in enumerate(test_dloader):
            # Extract corresponding latents
            z = jax.tree.map(lambda x: x[i*config.eval.batch_size:(i+1)*config.eval.batch_size], z_test_dataset)
            psnr = evaluate_batch(recon_enf_params, x, img, z)
            psnrs_evaluation.append(psnr)
        
        avg_psnr = sum(psnrs_evaluation) / len(psnrs_evaluation)
        
        # Visualization for first sample
        img, _ = next(iter(test_dloader))
        z = jax.tree.map(lambda x: x[:config.eval.batch_size], z_test_dataset)
        img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
        fig = plot_biobank_comparison(img[0], img_r[0], poses=z[0][0])
        
        logging.info(f"TEST ep {epoch} || mse: {sum(epoch_loss) / len(epoch_loss):.6f} || psnr: {avg_psnr:.4f}")
        wandb.log({"test-mse": sum(epoch_loss) / len(epoch_loss), "test-psnr": avg_psnr, "test-reconstruction": fig}, step=epoch)
        plt.close('all')
        
        # Save best results
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            logging.info(f"New best PSNR: {best_psnr:.4f}")
    
    logging.info(f"Final test PSNR: {best_psnr:.4f}")
    run.finish()


if __name__ == "__main__":
    app.run(main)
