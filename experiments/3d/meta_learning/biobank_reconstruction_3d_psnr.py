import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging
import time
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
import matplotlib.pyplot as plt

import wandb

# Custom imports
from experiments.datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

jax.config.update("jax_default_matmul_precision", "highest")

def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "biobank_reconstruction"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 3  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4

    config.recon_enf.num_latents = 512
    config.recon_enf.latent_dim = 64
    config.recon_enf.gaussian_window = True
    config.recon_enf.even_sampling = False
    config.recon_enf.latent_noise = False

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2
    config.dataset.z_indices = (0, 1)

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.)
    config.optim.inner_steps = 3
    
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 10
    config.train.log_interval = 25
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf"
    config.exp_name = "biobank_reconstruction"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
):
    """Plot original and reconstructed images for all slices in the z-axis.
    
    Args:
        original: Original images with shape (Z, H, W, C)
        reconstruction: Reconstructed images with shape (Z, H, W, C)
    """
    
    # NOTE: Clipping to to prevent warnings, NEWLY ADDED
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)
    
    num_slices = original.shape[0]  # Z is now the first dimension
    fig, axes = plt.subplots(num_slices, 2, figsize=(4, 2 * num_slices))
    fig.suptitle('Original (left) vs Reconstruction (right)')

    # Ensure axes is always a 2D array
    if num_slices == 1:
        axes = axes[None, :]  # Add an extra dimension to make it 2D

    for i in range(num_slices):
        # Clip to prevent warnings
        orig_slice = jnp.clip(original[i, :, :, 0], 0, 1)  # Use the first channel
        recon_slice = jnp.clip(reconstruction[i, :, :, 0], 0, 1)  # Use the first channel

        # Plot original
        axes[i, 0].imshow(orig_slice, cmap='gray')
        axes[i, 0].set_title(f'Original Slice {i}')

        # Plot reconstructed
        axes[i, 1].imshow(recon_slice, cmap='gray')
        axes[i, 1].set_title(f'Reconstruction Slice {i}')

        # Remove axes
        for ax in axes[i, :]:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('3d_biobank', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed, z_indices=list(config.dataset.z_indices))
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] # Image shape : (2, 128, 160, 1) : 40 960 datapoints 
    num_subsampled_points = img_shape[1] * img_shape[2]

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape, num_in=config.recon_enf.num_in)

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
        z_positions=len(list(config.dataset.z_indices)),
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x[:, :num_subsampled_points, :], *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)

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
            latent_noise=config.recon_enf.latent_noise,
        )

        def mse_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)
        
        def psnr(z):
            out = recon_enf.apply(enf_params, coords, *z)
            mse = jnp.mean((img - out) ** 2, axis=1) 
            
            # TODO: Check if max_pixel_value is correct
            max_pixel_value = 1.0 
            psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
            
            return jnp.mean(psnr)

        def inner_step(z, _):
            _, grads = jax.value_and_grad(mse_loss)(z)
            # Gradient descent update
            z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, grads, config.optim.inner_lr)
            return z, None
        
        # Perform inner loop optimization
        z, _ = jax.lax.scan(inner_step, z, None, length=config.optim.inner_steps)
        
        # Stop gradient if first order MAML
        if config.optim.first_order_maml:
            z = jax.lax.stop_gradient(z)
            
            
        psnr_value = psnr(z) # TODO: Do outside of loop 
        psnr_value = jax.lax.stop_gradient(psnr_value)
        
        
        return mse_loss(z), (z, psnr_value)

    @jax.jit
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, (z, psnr_value)), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z, psnr_value), enf_params, enf_opt_state, subkey

    @jax.jit
    def subsample_xy(key, x_, y_):
        """ Subsample coordinates and pixel values once.

        Args:
            key: PRNG key for random operations.
            x_: coordinates, shape (batch, num_points, 3)
            y_: pixel values, shape (batch, num_points, 1)
        
        Returns:
            x_i: Subsampled coordinates, shape (batch, num_subsampled_points, 3)
            y_i: Subsampled pixel values, shape (batch, num_subsampled_points, 1)
        """
        num_points = x_.shape[1]
        sub_mask = jax.random.permutation(key, num_points)[:num_subsampled_points]  # Get 3096 random indices
        x_i = x_[:, sub_mask]  # Index along the second dimension
        y_i = y_[:, sub_mask]  # Index along the second dimension
        return x_i, y_i
    
    
    def evaluate_test_set(enf_params, test_dloader, key):
        
        """Evaluate model on the entire test set."""
        psnrs = []
        mses = []
        
        img_r_slices = []
        img_slices = []
        
        
        # Loop over volumes in test_dloader
        for idx, (img, _) in enumerate(test_dloader):
            
            # Keep first volume for visualization
            if idx == 0:
                img_org = img
            
            psnr_patient = []
            mse_patient = []
                
            # Flatten input
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
                    
            # Loop over slices in volume
            for z_idx in range(img.shape[1]):
                
                # Get new key
                key, subkey = jax.random.split(key)
                                
                x_slice = x[:, (z_idx*num_subsampled_points):((z_idx+1)*num_subsampled_points), :]
                y_slice = y[:, (z_idx*num_subsampled_points):((z_idx+1)*num_subsampled_points), :]
                
                try: 
                    mse, (z, psnr_value) = recon_inner_loop(enf_params, x_slice, y_slice, key)
                except:
                    print("waddup")
                
                psnr_patient.append(psnr_value)
                mse_patient.append(mse)
                
                if idx == 0:
                    # Reshape to a single slice
                    img_r_slice = recon_enf.apply(enf_params, x_slice, *z).reshape(img[:,:1,:,:,:].shape)
                    img_r_slice = jnp.clip(img_r_slice, 0, 1) 
                    
                    img_slice = y_slice.reshape(img[:,:1,:,:,:].shape)
                
                    img_r_slices.append(img_r_slice) 
                    img_slices.append(img_slice)
                    
                
            psnrs.append(jnp.mean(jnp.array(psnr_patient)))
            mses.append(jnp.mean(jnp.array(mse_patient)))
            
        # Calculate average PSNR
        avg_psnr = jnp.mean(jnp.array(psnrs))
        avg_mse = jnp.mean(jnp.array(mses))
        
        img_r_slices = jnp.concatenate(img_r_slices, axis=1)
        img_slices = jnp.concatenate(img_slices, axis=1)
        
        return avg_psnr, avg_mse, img_slices, img_r_slices
    
    
    # Pretraining loop for fitting the ENF backbone
    glob_step = 0
    for epoch in range(config.train.num_epochs_train):
        epoch_loss = []
        epoch_psnr = []
        for i, (img, _) in enumerate(train_dloader):
                        
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1])) # Collapses (H,W, (Z))

            x_i, y_i = subsample_xy(key, x, y)
            
            # Perform outer loop optimization
            (loss, z, psnr_value), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x_i, y_i, recon_enf_params, recon_enf_opt_state, key)

            epoch_loss.append(loss)
            epoch_psnr.append(psnr_value)
            glob_step += 1
      
            if glob_step % config.train.log_interval == 0:
                
                eval_start_time = time.time()
                test_psnr, test_mse, img_slices, img_r_slices = evaluate_test_set(recon_enf_params, test_dloader, key)
                eval_duration = time.time() - eval_start_time
                logging.info(f"Test set evaluation time: {eval_duration:.2f} seconds")
                
                fig = plot_biobank_comparison(img_slices[0], img_r_slices[0])

                wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "test-psnr": test_psnr, "test-mse": test_mse, "reconstruction": fig}, step=glob_step)
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])} || test-psnr : {test_psnr} || test-mse: {test_mse}")
                
            
    run.finish()

if __name__ == "__main__":
    app.run(main)