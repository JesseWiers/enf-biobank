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
    config.run_name = "biobank_reconstruction"
    config.exp_name = "test"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 3  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 512
    config.recon_enf.latent_dim = 64
    config.recon_enf.z_positions = 2
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 50 
    config.dataset.num_patients_test = 10
    config.dataset.z_indices = (0, 1)

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4 
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.) # NOTE: Try 1e-3 
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 16
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 100
    config.train.log_interval = 200
    config.train_checkpoint_path = ''
    logging.getLogger().setLevel(logging.INFO)

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
    train_dloader, _ = get_dataloaders('3d_biobank_v2', config.train.batch_size, config.dataset.num_workers, 
                        num_train=config.dataset.num_patients_train, 
                        num_test=config.dataset.num_patients_test, 
                        seed=config.seed, 
                        shuffle_train=False,
                        z_indices=config.dataset.z_indices)
    
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] 
    num_subsampled_points = img_shape[1] * img_shape[2]
    
    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape, num_in=config.recon_enf.num_in)

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
        noise_scale=config.train.noise_scale,
        latent_noise=config.recon_enf.latent_noise,
        even_sampling=config.recon_enf.even_sampling,
        z_positions=config.recon_enf.z_positions,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x[:, :num_subsampled_points, :], *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)

    @jax.jit
    def train_step(coords, img, z, enf_params, enf_opt_state, key):

        def mse_loss(z, enf_params):  
            out = recon_enf.apply(enf_params, coords, *z)  
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)

        key, subkey = jax.random.split(key)
        loss, (z_grads, enf_grads) = jax.value_and_grad(mse_loss, argnums=(0, 1))(z, enf_params)
                
        z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, z_grads, config.optim.inner_lr)
        
        enf_grads, enf_opt_state = enf_opt.update(enf_grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)
                
        return (loss, z), enf_params, enf_opt_state, subkey

    @jax.jit
    def evaluate_batch(enf_params, coords, img, z):
        # Compute reconstruction
        img_r = recon_enf.apply(enf_params, coords, *z).reshape(img.shape)
        
        # Compute PSNR
        mse = jnp.mean((img - img_r) ** 2)
        max_pixel_value = 1.0 
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
        return psnr
    
    
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
    
    
    def evaluate_test_set(recon_enf_params, z_dataset, test_dloader, key):
        
        # Take first 500 volumes (first 10 patients)
        train_dloader_subset = itertools.islice(train_dloader, 500)
        
        psnrs = []
        img_r_slices = []
        img_slices = []
        
        for i, (img, _) in enumerate(train_dloader_subset):
            
            # Take z of patient i
            z = jax.tree.map(lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], z_dataset)
            
            if i == 0:
                img_org = img
                
            psnr_patient = []
            
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            for z_idx in range(img.shape[1]):
                
                key, subkey = jax.random.split(key)
                
                x_slice = x[:, (z_idx*num_subsampled_points):((z_idx+1)*num_subsampled_points), :]
                y_slice = y[:, (z_idx*num_subsampled_points):((z_idx+1)*num_subsampled_points), :]
                
                psnr_value = evaluate_batch(recon_enf_params, x_slice, y_slice, z)
                psnr_patient.append(psnr_value)
                
                if i==0: 
                    img_r_slice = recon_enf.apply(recon_enf_params, x_slice, *z).reshape(img[:,:1,:,:,:].shape)
                    img_r_slice = jnp.clip(img_r_slice, 0, 1) 
                    
                    img_slice = y_slice.reshape(img[:,:1,:,:,:].shape)
                    
                    img_r_slices.append(img_r_slice) 
                    img_slices.append(img_slice)
                
            psnrs.append(jnp.mean(jnp.array(psnr_patient)))
            
        # Calculate average PSNR
        avg_psnr = jnp.mean(jnp.array(psnrs))
        
        img_r_slices = jnp.concatenate(img_r_slices, axis=1)
        img_slices = jnp.concatenate(img_slices, axis=1)
        
        return avg_psnr, img_slices, img_r_slices
                    
                      
    if os.path.exists(config.train_checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {config.train_checkpoint_path}\033[0m")
        recon_enf_params, recon_enf_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(config.train_checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
    
    
    # Training loop for fitting the ENF backbone
    num_samples = len(train_dloader) * config.train.batch_size
    logging.info(f"Initializing latens for {num_samples} samples")
       
    z_dataset = initialize_latents(
                    batch_size=num_samples,
                    num_latents=config.recon_enf.num_latents,
                    latent_dim=config.recon_enf.latent_dim,
                    data_dim=config.recon_enf.num_in,
                    bi_invariant_cls=TranslationBI,
                    key=key,  
                )

    glob_step = 0
    best_psnr = float('-inf')
    
    # Training loop
    for epoch in range(config.train.num_epochs_train):

        epoch_loss = []
        
        start_time = time.time()

        for i, (img, _) in enumerate(train_dloader):
            
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            x_i, y_i = subsample_xy(key, x, y)
                        
            # Extract latents corresponding to batch (NOTE: MIGHT BE QUITE INEFFICIENT TO GO OVER ALL LATENTS)
            z = jax.tree.map(lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], z_dataset)
        
            (loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                x_i, y_i, z, recon_enf_params, recon_enf_opt_state, key)
            
            # Update dataset with new latents 
            z_dataset = jax.tree.map(lambda full, partial: full.at[i*config.train.batch_size:(i+1)*config.train.batch_size].set(partial), z_dataset, z)

            epoch_loss.append(loss)
            glob_step += 1
            
        end_time = time.time()
        logging.info(f"Epoch {epoch} took {end_time - start_time} seconds")
        
        
        # Evaluation    
        start_time = time.time()
        test_psnr, img_slices, img_r_slices = evaluate_test_set(recon_enf_params, z_dataset, train_dloader, key)
        end_time = time.time()
        logging.info(f"Evaluation took {end_time - start_time} seconds")
        
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            logging.info(f"New best PSNR: {best_psnr}")
            save_checkpoint(f"checkpoints/{config.exp_name}/{config.run_name}", recon_enf_params, recon_enf_opt_state, epoch, glob_step, best_psnr)
            
        fig = plot_biobank_comparison(img_slices[0], img_r_slices[0])
        wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "test-psnr": test_psnr, "epoch": epoch, "reconstruction": fig}, step=glob_step)
        plt.close('all')
        
    run.finish()


if __name__ == "__main__":
    app.run(main)
