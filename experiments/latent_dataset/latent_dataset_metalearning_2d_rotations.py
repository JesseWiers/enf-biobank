import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

import wandb

import os
import pickle

from flax.training import checkpoints
import flax
flax.training.checkpoints.CHECKPOINT_GDA = False

import h5py

# Custom imports
from experiments.datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier

jax.config.update("jax_default_matmul_precision", "highest")

    
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
    config.run_name = "biobank_reconstruction"
    config.exp_name = "test"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 128
    config.recon_enf.num_in = 2  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (3.0, 5.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 64
    config.recon_enf.latent_dim = 16
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True
    
    config.recon_enf.checkpoint_path = ""

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 1
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 10
    config.train.log_interval = 50

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (2., 30., 0.) # (pose, context, window), orginally (2., 30., 0.)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False
    
    config.dataset = ml_collections.ConfigDict()
    config.dataset.root = "/projects/prjs1252/data_jesse/cmr_cropped"
    config.dataset.latent_dataset_path = "/projects/prjs1252/data_jesse/latent_dataset_4d.h5"

    logging.getLogger().setLevel(logging.INFO)

    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def rotate_image(image, angle_degrees):
    """Rotate image by 90, 180 or 270 degrees"""
    k = angle_degrees // 90  # k=1 for 90°, k=2 for 180°, k=3 for 270°
    return np.rot90(image, k=k)

def main(_):

    # Get config
    config = _CONFIG.value
    
        
    patient_paths = os.listdir(config.dataset.root)
    
    for patient_path in patient_paths:

        nifti_file_path = os.path.join(config.dataset.root, patient_path, "cropped_sa.nii.gz")
        
        nifti_image = nib.load(nifti_file_path)
        image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                        
        H, W, Z, T = image_data.shape
            
        # Iterate over timesteps
        for timestep in range(T):
            for slice_number in range(Z):
                image = image_data[:, :, slice_number, timestep] # Load image 
                        
                # Min-max normalization per slice
                slice_min = np.min(image)
                slice_max = np.max(image)
                
                image = (image - slice_min) / (slice_max - slice_min)
            
                # Assert all values are between 0 and 1
                assert np.all(image >= 0) and np.all(image <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={timestep}, z={slice_number}"
            
                sample_img = image[..., np.newaxis] # Add axis
                
                break
            
            break
        
        break
                
    img_shape = sample_img[:,:,0].shape

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

    # config.recon_enf.checkpoint_path = '/home/jwiers/deeprisk/new_codebase/enf-biobank/checkpoints/train_4500_test_20_latents_64_16_batchsize_1/checkpoint_167400.pkl'
    
    if os.path.exists(config.recon_enf.checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {config.recon_enf.checkpoint_path}\033[0m")
        recon_enf_params, recon_enf_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(config.recon_enf.checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
        
    output_dir = os.path.dirname(config.dataset.latent_dataset_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    hdf5_path = config.dataset.latent_dataset_path
    
    with h5py.File(hdf5_path, 'a') as f:
        # Create a group for patient metadata if it doesn't exist
        if "metadata" not in f:
            meta_group = f.create_group("metadata")
            meta_group.attrs['num_latents'] = config.recon_enf.num_latents
            meta_group.attrs['latent_dim'] = config.recon_enf.latent_dim
            meta_group.attrs['checkpoint_path'] = config.recon_enf.checkpoint_path
        else:
            meta_group = f["metadata"]
        
        # Create a group for patients if it doesn't exist
        if "patients" not in f:
            patients_group = f.create_group("patients")
        else:
            patients_group = f["patients"]
        
        # Process all patients
        for patient_idx, patient_path in enumerate(patient_paths):
            # Check if this patient is already in the dataset
            if patient_path in patients_group:
                logging.info(f"Patient {patient_path} already exists in the dataset. Skipping.")
                continue
            
            nifti_file_path = os.path.join(config.dataset.root, patient_path, "cropped_sa.nii.gz")
            
            try:
                nifti_image = nib.load(nifti_file_path)
                image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                                
                H, W, Z, T = image_data.shape
                
                z_linspaces = jnp.linspace(-1, 1, Z)
                t_linspaces = jnp.linspace(-1, 1, T)
                
                # Lists for original version
                all_p = []
                all_c = []
                all_g = []
                
                # Lists for rotated version
                all_p_rotated = []
                all_c_rotated = []
                all_g_rotated = []
                
                # Randomly select rotation angle for this volume
                key, rotation_key = jax.random.split(key)
                rotation_angles = [90, 180, 270]
                random_rotation = rotation_angles[jax.random.randint(rotation_key, (), 0, 3)]
                
                # Iterate over timesteps
                for timestep in range(T):
                    for slice_number in range(Z):
                        image = image_data[:, :, slice_number, timestep] # Load image 
                        
                        # Process original image
                        slice_min = np.min(image)
                        slice_max = np.max(image)
                        image_orig = (image - slice_min) / (slice_max - slice_min)
                        assert np.all(image_orig >= 0) and np.all(image_orig <= 1)
                        img_orig = image_orig[..., np.newaxis]
                        y_orig = jnp.reshape(img_orig, (1, -1, img_orig.shape[-1]))
                        
                        # Process rotated image
                        image_rotated = rotate_image(image, random_rotation)
                        slice_min = np.min(image_rotated)
                        slice_max = np.max(image_rotated)
                        image_rotated = (image_rotated - slice_min) / (slice_max - slice_min)
                        assert np.all(image_rotated >= 0) and np.all(image_rotated <= 1)
                        img_rotated = image_rotated[..., np.newaxis]
                        y_rotated = jnp.reshape(img_rotated, (1, -1, img_rotated.shape[-1]))
                        
                        # Get latents for original image
                        key, subkey = jax.random.split(key)
                        _, (z_orig, _) = recon_inner_loop(recon_enf_params, x, y_orig, subkey)
                        p_orig, c_orig, g_orig = z_orig
                        
                        # Get latents for rotated image
                        key, subkey = jax.random.split(key)
                        _, (z_rotated, _) = recon_inner_loop(recon_enf_params, x, y_rotated, subkey)
                        p_rotated, c_rotated, g_rotated = z_rotated
                        
                        # Get normalized coordinates
                        normalized_z_coord = z_linspaces[slice_number]
                        normalized_t_coord = t_linspaces[timestep]
                        
                        # Process original latents
                        t_coords = jnp.ones((p_orig.shape[0], p_orig.shape[1], 1)) * normalized_t_coord
                        z_coords = jnp.ones((p_orig.shape[0], p_orig.shape[1], 1)) * normalized_z_coord
                        new_p = jnp.concatenate([t_coords, z_coords, p_orig], axis=2)
                        
                        # Process rotated latents
                        t_coords_rotated = jnp.ones((p_rotated.shape[0], p_rotated.shape[1], 1)) * normalized_t_coord
                        z_coords_rotated = jnp.ones((p_rotated.shape[0], p_rotated.shape[1], 1)) * normalized_z_coord
                        new_p_rotated = jnp.concatenate([t_coords_rotated, z_coords_rotated, p_rotated], axis=2)
                        
                        # Append all latents
                        all_p.append(new_p)
                        all_c.append(c_orig)
                        all_g.append(g_orig)
                        
                        all_p_rotated.append(new_p_rotated)
                        all_c_rotated.append(c_rotated)
                        all_g_rotated.append(g_rotated)
                
                # Concatenate all latent components
                combined_p = jnp.concatenate(all_p, axis=1)
                combined_c = jnp.concatenate(all_c, axis=1)
                combined_g = jnp.concatenate(all_g, axis=1)
                
                combined_p_rotated = jnp.concatenate(all_p_rotated, axis=1)
                combined_c_rotated = jnp.concatenate(all_c_rotated, axis=1)
                combined_g_rotated = jnp.concatenate(all_g_rotated, axis=1)
                
                # Create a group for this patient
                patient_group = patients_group.create_group(patient_path)
                
                # Store image shape information and rotation metadata
                patient_group.attrs['H'] = H
                patient_group.attrs['W'] = W
                patient_group.attrs['Z'] = Z
                patient_group.attrs['T'] = T
                patient_group.attrs['rotation_angle'] = random_rotation
                
                # Store both original and rotated latent variables
                patient_group.create_dataset('p', data=np.array(combined_p))
                patient_group.create_dataset('c', data=np.array(combined_c))
                patient_group.create_dataset('g', data=np.array(combined_g))
                
                patient_group.create_dataset('p_rotated', data=np.array(combined_p_rotated))
                patient_group.create_dataset('c_rotated', data=np.array(combined_c_rotated))
                patient_group.create_dataset('g_rotated', data=np.array(combined_g_rotated))
                
                logging.info(f"Processed patient {patient_idx+1}/{len(patient_paths)}: {patient_path} with rotation {random_rotation}°")
                
            except Exception as e:
                logging.error(f"Error processing patient {patient_path}: {str(e)}")
                continue
            
        logging.info(f"Dataset saved to {hdf5_path}")


if __name__ == "__main__":
    app.run(main)
