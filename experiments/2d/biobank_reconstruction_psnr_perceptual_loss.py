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
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

from src.perceptual_loss import VGGPerceptualLoss

jax.config.update("jax_default_matmul_precision", "highest")


def save_checkpoint(checkpoint_dir, model_params, optimizer_state, epoch, global_step, best_perceptual_loss):
    """
    Saves the model parameters, optimizer state, and training metadata.
    
    Args:
        checkpoint_dir (str): Directory where the checkpoint will be saved.
        model_params (PyTree): The parameters of the model to be saved.
        optimizer_state (PyTree): The state of the optimizer.
        epoch (int): Current epoch number.
        global_step (int): Current training step.
        best_perceptual_loss (float): The best perceptual loss value encountered so far.
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
        "best_perceptual_loss": best_perceptual_loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pkl")
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
        
    print(f"Checkpoint saved at step {global_step} in {checkpoint_dir}")

def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "biobank_reconstruction_perceptual"
    config.exp_name = "test_perceptual"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 2  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (3.0, 5.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 16
    config.recon_enf.latent_dim = 64
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (2., 30., 0.) # (pose, context, window), orginally (2., 30., 0.)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 10
    config.train.log_interval = 50
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf_perceptual"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
):
    """Plot original and reconstructed biobank images side by side.
    
    Args:
        original: Original images with shape (H, W, 3)
        reconstruction: Reconstructed images with shape (H, W, 3)
        poses: Optional poses to plot on the image
    """
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    fig.suptitle('Original vs Perceptual Reconstruction')
    
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
    train_dloader, test_dloader = get_dataloaders('2d_biobank', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] # Image shape : (128, 160) : 20 480 datapoints

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)

    # Define the reconstruction and segmentation models
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
        noise_scale=config.train.noise_scale,
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)

    # Initialize perceptual loss model
    perceptual_loss_fn = VGGPerceptualLoss(seed=config.seed)
    
    # Create a dummy input to initialize VGG parameters
    dummy_input = jnp.zeros((1, 224, 224, 3))
    vgg_params = perceptual_loss_fn.init_params(dummy_input)

    @jax.jit
    def recon_inner_loop(enf_params, coords, img, key, vgg_params):
        z = initialize_latents(
            batch_size=config.train.batch_size,
            num_latents=config.recon_enf.num_latents,
            latent_dim=config.recon_enf.latent_dim,
            data_dim=config.recon_enf.num_in,
            bi_invariant_cls=TranslationBI,
            key=key,
            noise_scale=config.train.noise_scale,
            latent_noise=config.recon_enf.latent_noise,
        )

        def perceptual_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            
            # Reshape for perceptual loss calculation
            out_img = out.reshape((-1,) + img_shape + (1,))
            target_img = img.reshape((-1,) + img_shape + (1,))
            
            # If images are grayscale, duplicate to 3 channels for VGG
            if out_img.shape[-1] == 1:
                out_img_rgb = jnp.repeat(out_img, 3, axis=-1)
                target_img_rgb = jnp.repeat(target_img, 3, axis=-1)
            else:
                out_img_rgb = out_img
                target_img_rgb = target_img
            
            # Calculate perceptual loss with explicit parameters
            return perceptual_loss_fn.apply(vgg_params, out_img_rgb, target_img_rgb)
        
        def mse(z):
            out = recon_enf.apply(enf_params, coords, *z)
            return jnp.mean((img - out) ** 2, axis=(1, 2))
            
        def psnr(z):
            mse_values = mse(z)
            max_pixel_value = 1.0 
            psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse_values))
            return jnp.mean(psnr)

        def inner_step(z, _):
            _, grads = jax.value_and_grad(perceptual_loss)(z)
            # Gradient descent update
            z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, grads, config.optim.inner_lr)
            return z, None
        
        # Perform inner loop optimization
        z, _ = jax.lax.scan(inner_step, z, None, length=config.optim.inner_steps)
        
        # Stop gradient if first order MAML
        if config.optim.first_order_maml:
            z = jax.lax.stop_gradient(z)
        
        # Calculate all metrics
        perc_loss = perceptual_loss(z)
        psnr_value = psnr(z)
        mse_value = jnp.mean(mse(z))
        
        # Stop gradients for evaluation metrics
        psnr_value = jax.lax.stop_gradient(psnr_value)
        mse_value = jax.lax.stop_gradient(mse_value)
        
        return perc_loss, (z, psnr_value, mse_value, perc_loss)

    @jax.jit
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key, vgg_params):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, (z, psnr_value, mse_value, perc_loss)), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key, vgg_params)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z, psnr_value, mse_value, perc_loss), enf_params, enf_opt_state, subkey
    
    
    def evaluate_test_set(enf_params, test_dloader, key, vgg_params):
        """Evaluate model on the entire test set."""
        psnrs = []
        mses = []
        perceptual_losses = []
        reconstructions = []
        originals = []
        
        for img, _ in test_dloader:
            
            # Flatten input
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Get new key
            key, subkey = jax.random.split(key)
            
            perc_loss, (z, psnr_value, mse_value, p_loss) = recon_inner_loop(enf_params, x, y, key, vgg_params)
            
            psnrs.append(psnr_value)
            mses.append(mse_value)
            perceptual_losses.append(p_loss)
            img_r = recon_enf.apply(enf_params, x, *z).reshape(img.shape)
            
            originals.append(img)
            reconstructions.append(img_r)
                 
        # Calculate average metrics
        avg_psnr = jnp.mean(jnp.array(psnrs))
        avg_mse = jnp.mean(jnp.array(mses))
        avg_perceptual_loss = jnp.mean(jnp.array(perceptual_losses))
        
        # Return first batch for visualization
        return avg_psnr, avg_mse, avg_perceptual_loss, originals[0], reconstructions[0]

 
    # Pretraining loop for fitting the ENF backbone
    best_perceptual_loss = float('inf')
    glob_step = 0
    for epoch in range(config.train.num_epochs_train):
        epoch_perceptual_loss = []
        epoch_psnr = []
        epoch_mse = []
        for i, (img, _) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Pass VGG params to outer step
            (loss, z, psnr_value, mse_value, perc_loss), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x, y, recon_enf_params, recon_enf_opt_state, key, vgg_params)

            epoch_perceptual_loss.append(perc_loss)
            epoch_psnr.append(psnr_value)
            epoch_mse.append(mse_value)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Evaluate on test set
                test_psnr, test_mse, test_perceptual, img, img_r = evaluate_test_set(recon_enf_params, test_dloader, key, vgg_params)
                
                # Save checkpoint based on perceptual loss (lower is better)
                if test_perceptual < best_perceptual_loss:
                    best_perceptual_loss = test_perceptual
                    save_checkpoint(f"checkpoints/{config.run_name}", recon_enf_params, recon_enf_opt_state, epoch, glob_step, best_perceptual_loss)

                fig = plot_biobank_comparison(img[0], img_r[0], poses=z[0][0])
                
                # Log all metrics
                wandb.log({
                    "train-perceptual-loss": sum(epoch_perceptual_loss) / len(epoch_perceptual_loss), 
                    "train-mse": sum(epoch_mse) / len(epoch_mse),
                    "train-psnr": sum(epoch_psnr) / len(epoch_psnr),
                    "test-perceptual-loss": test_perceptual,
                    "test-mse": test_mse, 
                    "test-psnr": test_psnr, 
                    "reconstruction": fig
                }, step=glob_step)
                
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || "
                             f"perceptual: {sum(epoch_perceptual_loss[-10:]) / len(epoch_perceptual_loss[-10:])} || "
                             f"test-perceptual: {test_perceptual} || "
                             f"test-mse: {test_mse} || "
                             f"test-psnr: {test_psnr}")

    run.finish()


if __name__ == "__main__":
    app.run(main)
