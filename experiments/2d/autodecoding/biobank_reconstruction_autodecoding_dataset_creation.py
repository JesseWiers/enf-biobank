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
import h5py
import numpy as np
# Custom imports
from experiments.datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

import nibabel as nib

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

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "latent_dataset_creation_autodecoding"
    config.exp_name = "test"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 128
    config.recon_enf.num_in = 2  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True
    
    config.recon_enf.num_latents = 128
    config.recon_enf.latent_dim = 64
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True
    config.recon_enf.checkpoint_path = "/home/jwiers/deeprisk/new_codebase/enf-biobank/model_checkpoints_autodecoding/23-06_autodecoding_2d/50_patients_lr_01_multq_30_multv_60_6_hours_128l_64d/checkpoint_1839753.pkl"  # Path to trained model checkpoint
    
    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.inner_lr = (0., 0.1, 0.) # (pose, context, window), orginally (2., 30., 0.) # NOTE: Try 1e-3 
    config.optim.num_epochs = 25  # Number of epochs to optimize latents per patient
    
    config.train = ml_collections.ConfigDict()
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    
    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.root = "/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped"  # Root directory containing patient data
    config.dataset.num_patients_train = 50
    config.dataset.latent_dataset_path = f"/projects/prjs1252/data_jesse_v2/latent_dataset_autodecoding_{config.recon_enf.num_latents}l_{config.recon_enf.latent_dim}d_{config.dataset.num_patients_train}_{config.optim.num_epochs}_epochs_final_patients.h5"  # Where to save the latent dataset
    
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


def plot_slice_reconstruction(original, reconstruction, patient_id, slice_idx, timestep):
    """Create figure for wandb logging of original and reconstructed slice side by side.
    
    Args:
        original: Original slice with shape (H, W)
        reconstruction: Reconstructed slice with shape (H, W)
        patient_id: Patient identifier for the title
        slice_idx: Z-axis slice index
        timestep: Time point index
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Patient {patient_id} - Slice {slice_idx}, Time {timestep}')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Plot reconstructed
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Get list of patient directories
    patient_paths = os.listdir(config.dataset.root)

    # Setup model and coordinate grid
    key = jax.random.PRNGKey(config.seed)
    
    # Get a sample image to determine shape
    for patient_path in patient_paths:
        nifti_file_path = os.path.join(config.dataset.root, patient_path, "cropped_sa.nii.gz")
        nifti_image = nib.load(nifti_file_path)
        image_data = nifti_image.get_fdata()
        sample_img = image_data[:, :, 0, 0][..., np.newaxis]
        break

    x = create_coordinate_grid(batch_size=1, img_shape=sample_img.shape[:2])
    
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

    # Load trained model parameters
    if not os.path.exists(config.recon_enf.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.recon_enf.checkpoint_path}")
    
    recon_enf_params, _, _, _, _ = load_checkpoint(config.recon_enf.checkpoint_path)

    @jax.jit
    def optimize_latent_step(coords, img, z, key):
        def mse_loss(z):  
            out = recon_enf.apply(recon_enf_params, coords, *z)  
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)

        loss, z_grads = jax.value_and_grad(mse_loss)(z)
        z = jax.tree_map(lambda z, grad, lr: z - lr * grad, z, z_grads, config.optim.inner_lr)
        
        # Calculate PSNR
        reconstruction = recon_enf.apply(recon_enf_params, coords, *z)
        mse = jnp.mean((img - reconstruction) ** 2)
        max_pixel_value = 1.0
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
        
        return loss, psnr, z

    # Create/open HDF5 file
    with h5py.File(config.dataset.latent_dataset_path, 'a') as f:
        
        # Create metadata group if it doesn't exist
        if "metadata" not in f:
            meta_group = f.create_group("metadata")
            meta_group.attrs['num_latents'] = config.recon_enf.num_latents
            meta_group.attrs['latent_dim'] = config.recon_enf.latent_dim
            meta_group.attrs['checkpoint_path'] = config.recon_enf.checkpoint_path
        
        # Create patients group if it doesn't exist
        if "patients" not in f:
            patients_group = f.create_group("patients")
        else:
            patients_group = f["patients"]

        # Process each patient
        for patient_idx, patient_path in enumerate(patient_paths):
            
            logging.info(f"Processing patient {patient_idx}")
            
            # time operation
            start_time = time.time()
        
            # Skip if patient already exists in dataset
            if patient_path in patients_group:
                logging.info(f"Patient {patient_path} already exists. Skipping.")
                continue

            try:
                # Load patient data
                nifti_file_path = os.path.join(config.dataset.root, patient_path, "cropped_sa.nii.gz")
                nifti_image = nib.load(nifti_file_path)
                image_data = nifti_image.get_fdata()  # Shape:  [H, W, Z, T]
                H, W, Z, T = image_data.shape

                # Create normalized coordinate grids for z and t dimensions
                z_linspace = jnp.linspace(-1, 1, Z)
                t_linspace = jnp.linspace(-1, 1, T)

                # Initialize latents for entire volume
                num_slices = Z * T
                z_volume = initialize_latents(
                    batch_size=num_slices,
                    num_latents=config.recon_enf.num_latents,
                    latent_dim=config.recon_enf.latent_dim,
                    data_dim=config.recon_enf.num_in,
                    bi_invariant_cls=TranslationBI,
                    key=key,
                    noise_scale=config.train.noise_scale,
                    even_sampling=config.recon_enf.even_sampling,
                    latent_noise=config.recon_enf.latent_noise,
                )

                # Optimize latents for all slices over multiple epochs
                for epoch in range(config.optim.num_epochs):
                    epoch_loss = []
                    epoch_psnr = []
                    
                    # Process each slice
                    for t in range(T):
                        for z in range(Z):
                            slice_idx = t * Z + z
                            
                            # Get and normalize image slice
                            image = image_data[:, :, z, t]
                            slice_min, slice_max = np.min(image), np.max(image)
                            image = (image - slice_min) / (slice_max - slice_min)
                            image = image[..., np.newaxis]
                            
                            # Reshape for model
                            y = jnp.reshape(image, (1, -1, 1))
                            
                            # Get current latents for this slice
                            z_slice = jax.tree_map(lambda x: x[slice_idx:slice_idx+1], z_volume)
                            
                            # Optimize latents
                            loss, psnr, updated_z = optimize_latent_step(x, y, z_slice, key)
                            
                            # Update volume latents
                            z_volume = jax.tree_map(
                                lambda full, partial: full.at[slice_idx:slice_idx+1].set(partial),
                                z_volume, updated_z
                            )
                            
                            epoch_loss.append(loss)
                            epoch_psnr.append(psnr)
                    
                    avg_loss = sum(epoch_loss) / len(epoch_loss)
                    avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
                    # logging.info(f"Patient {patient_path}, Epoch {epoch}: Average loss = {avg_loss:.6f}, Average PSNR = {avg_psnr:.2f}")
                    
                    # Log to wandb
                    wandb.log({
                        f"patient_{patient_path}_loss": avg_loss,
                        f"patient_{patient_path}_psnr": avg_psnr,
                        "epoch": epoch
                    })

                    # logging.info(f"Patient {patient_path}, Epoch {epoch}: Average loss = {avg_loss:.6f}, Average PSNR = {avg_psnr:.2f}")

                # Create patient group and save metadata
                patient_group = patients_group.create_group(patient_path)
                patient_group.attrs['H'] = H
                patient_group.attrs['W'] = W
                patient_group.attrs['Z'] = Z
                patient_group.attrs['T'] = T

                # Reshape and save latents
                p, c, g = z_volume
                
                # Add t and z coordinates to poses
                all_poses = []
                for t_idx in range(T):
                    for z_idx in range(Z):
                        slice_idx = t_idx * Z + z_idx
                        poses = p[slice_idx]
                        
                        # Create coordinate arrays
                        t_coords = jnp.ones((poses.shape[0], 1)) * t_linspace[t_idx]
                        z_coords = jnp.ones((poses.shape[0], 1)) * z_linspace[z_idx]
                        
                        # Concatenate coordinates
                        extended_poses = jnp.concatenate([
                            t_coords, z_coords, poses
                        ], axis=1)
                        
                        all_poses.append(extended_poses)

                # Save latent components
                patient_group.create_dataset('p', data=np.array(jnp.concatenate(all_poses, axis=0)))
                patient_group.create_dataset('c', data=np.array(jnp.reshape(c, (-1, c.shape[-1]))))
                patient_group.create_dataset('g', data=np.array(jnp.reshape(g, (-1, g.shape[-1]))))

                logging.info(f"Saved latents for patient {patient_path}")

                # After optimization epochs for a patient, visualize a middle slice
                mid_z = Z // 2
                mid_t = T // 2
                slice_idx = mid_t * Z + mid_z
                
                # Get and normalize the original middle slice
                original_slice = image_data[:, :, mid_z, mid_t]
                slice_min, slice_max = np.min(original_slice), np.max(original_slice)
                original_slice = (original_slice - slice_min) / (slice_max - slice_min)
                
                # Get latents for this slice
                z_slice = jax.tree_map(lambda x: x[slice_idx:slice_idx+1], z_volume)
                
                try:
                    # Generate reconstruction
                    reconstruction = recon_enf.apply(
                        recon_enf_params, 
                        x, 
                        *z_slice
                    )
                    
                    # Reshape correctly: from (1, H*W, 1) to (H, W)
                    reconstruction = reconstruction.reshape(1, H, W, 1)[0, :, :, 0]
                    
                    # Create visualization and log to wandb
                    fig = plot_slice_reconstruction(
                        original_slice, 
                        reconstruction, 
                        patient_path, 
                        mid_z, 
                        mid_t
                    )
                    wandb.log({
                        f"patient_{patient_path}_reconstruction": wandb.Image(fig),
                        "epoch": config.optim.num_epochs - 1  # Log at final epoch
                    })
                    plt.close(fig)  # Clean up the figure
                    logging.info(f"Logged reconstruction visualization for patient {patient_path}")

                except Exception as e:
                    logging.warning(f"Failed to visualize reconstruction for patient {patient_path}: {str(e)}")
                    

                logging.info(f"Patient {patient_path}, Epoch {epoch}: Average loss = {avg_loss:.6f}, Average PSNR = {avg_psnr:.2f}")
                # time operation
                end_time = time.time()
                logging.info(f"Time taken for patient {patient_path}: {end_time - start_time} seconds")

            except Exception as e:
                logging.error(f"Error processing patient {patient_path}: {str(e)}")
                continue

    logging.info(f"Latent dataset saved to {config.dataset.latent_dataset_path}")

    run.finish()


if __name__ == "__main__":
    app.run(main)
