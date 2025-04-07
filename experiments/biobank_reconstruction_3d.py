import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging

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
    config.recon_enf.num_in = 3  # Images are 2D
    config.recon_enf.num_out = 1  # 3 channels
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4

    config.recon_enf.num_latents = 128
    config.recon_enf.latent_dim = 64

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2
    config.dataset.z_indices = (0,1,2,3,4)

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 2
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_pretrain = 10
    config.train.log_interval = 100
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
):
    """Plot original and reconstructed images for all slices in the z-axis.
    
    Args:
        original: Original images with shape (H, W, D, C)
        reconstruction: Reconstructed images with shape (H, W, D, C)
        poses: Optional poses to plot on the image
    """
    num_slices = original.shape[2]  # Assuming the third dimension is the z-axis
    fig, axes = plt.subplots(num_slices, 3, figsize=(6, 2 * num_slices))
    fig.suptitle('Original (left) vs Reconstruction (middle)')

    for i in range(num_slices):
        # Clip to prevent warnings
        orig_slice = jnp.clip(original[:, :, i, 0], 0, 1)  # Use the first channel
        recon_slice = jnp.clip(reconstruction[:, :, i, 0], 0, 1)  # Use the first channel

        # Plot original
        axes[i, 0].imshow(orig_slice, cmap='gray')
        axes[i, 0].set_title(f'Original Slice {i}')

        # Plot reconstructed
        axes[i, 1].imshow(recon_slice, cmap='gray')
        axes[i, 1].set_title(f'Reconstruction Slice {i}')

        # Plot poses
        if poses is not None:
            # Map to 0-W range
            pose_slice = poses[i]
            pose_slice = (pose_slice + 1) * orig_slice.shape[0] / 2
            axes[i, 2].imshow(recon_slice, cmap='gray')
            axes[i, 2].scatter(pose_slice[:, 0], pose_slice[:, 1], c='r', s=2)
            axes[i, 2].set_title(f'Poses Slice {i}')

        # Remove axes
        for ax in axes[i, :]:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project="enf-min-bio-scratch-3d", config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)


    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('3d_biobank', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed, z_indices=list(config.dataset.z_indices))
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] # Image shape : (128, 160, 2) : 40 960 datapoints
         
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
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)

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
        )

        def mse_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)

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
        return mse_loss(z), z

    @jax.jit
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, z), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z), enf_params, enf_opt_state, subkey
    
    
    # Pretraining loop for fitting the ENF backbone
    glob_step = 0
    for epoch in range(config.train.num_epochs_pretrain):
        epoch_loss = []
        for i, (img, _) in enumerate(train_dloader):
                        
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Perform outer loop optimization
            (loss, z), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x, y, recon_enf_params, recon_enf_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                img_r = recon_enf.apply(recon_enf_params, x, *z)
                                
                img_r = img_r.reshape(img.shape)
                
                print("img[0] shape: ", img[0].shape)
                print("img_r[0] shape: ", img_r[0].shape)
                
                # Note: Currently passes first image in batch and first slice in z dimension
                fig = plot_biobank_comparison(img[0], img_r[0])

                wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "reconstruction": fig}, step=glob_step)
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])}")

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
