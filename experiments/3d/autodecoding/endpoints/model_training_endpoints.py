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
    config.run_name = "biobank_reconstruction_3d_lvef_autodecoding"
    config.exp_name = "test"

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 4  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 1296 
    config.recon_enf.latent_dim = 32
    config.recon_enf.z_positions = 2
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 100
    config.dataset.num_patients_test = 10
    config.dataset.z_indices = (0, 1, 2, 3, 4, 5, 6, 7)
    config.dataset.t_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4 
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.) # NOTE: Try 1e-3 
    

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 100
    config.train.log_interval = 50
    config.train.num_subsampled_points = None  # Will be set based on image shape
    logging.getLogger().setLevel(logging.INFO)

    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())

def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    t_idx: int,  # Add t_idx as a parameter
    t_indices: tuple,  # Add t_indices to show actual timepoint
):
    """Plot original and reconstructed images for all z-slices at the pre-selected time point.
    
    Args:
        original: Original images with shape (Z, H, W, 1)
        reconstruction: Reconstructed images with shape (Z, H, W, 1)
        t_idx: Index in t_indices being plotted
        t_indices: Full list of time indices being used
    """
    num_slices = original.shape[0]  # Number of Z slices
    fig, axes = plt.subplots(num_slices, 2, figsize=(4, 2 * num_slices))
    fig.suptitle(f'Original (left) vs Reconstruction (right) at t={t_indices[t_idx]}')

    # Ensure axes is always 2D array
    if num_slices == 1:
        axes = axes[None, :]

    for i in range(num_slices):
        # Plot original
        axes[i, 0].imshow(original[i, :, :, 0], cmap='gray')
        axes[i, 0].set_title(f'Original Slice {i}')

        # Plot reconstructed
        axes[i, 1].imshow(reconstruction[i, :, :, 0], cmap='gray')
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
    train_dloader, _ = get_dataloaders(
        'endpoints_4d', 
        config.train.batch_size, 
        config.dataset.num_workers, 
        num_train=config.dataset.num_patients_train, 
        num_test=config.dataset.num_patients_test, 
        seed=config.seed, 
        z_indices=config.dataset.z_indices,
        t_indices=config.dataset.t_indices,
        shuffle_train=False
    )
    sample_img = next(iter(train_dloader))
    img_shape = sample_img[0].shape[1:] # [T, Z, H, W]

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(
        batch_size=config.train.batch_size, 
        img_shape=img_shape,
        num_in=config.recon_enf.num_in  # 3D coordinates
    ) # [T * Z * H * W]

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
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
        z_positions=config.recon_enf.z_positions,
    )

    num_subsampled_points = img_shape[1] * img_shape[2] * img_shape[3]  # H * W for each slice

    # Init the model
    recon_enf_params = recon_enf.init(key, x[:, :num_subsampled_points, :], *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)


    @jax.jit
    def subsample_xy(key, x_, y_):
        """Subsample coordinates and pixel values.

        Args:
            key: PRNG key
            x_: coordinates, shape (batch, num_points, 3)
            y_: pixel values, shape (batch, num_points, 1)
        
        Returns:
            x_i: Subsampled coordinates (batch, num_subsampled_points, 3)
            y_i: Subsampled pixel values (batch, num_subsampled_points, 1)
        """
        num_points = x_.shape[1]
        sub_mask = jax.random.permutation(key, num_points)[:num_subsampled_points]
        x_i = x_[:, sub_mask]
        y_i = y_[:, sub_mask]
        return x_i, y_i

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
        
        # Compute PSNR per slice and average
        mse = jnp.mean((img - img_r) ** 2, axis=(1, 2, 4))  # Average over H, W, channels
        max_pixel_value = 1.0
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
        return jnp.mean(psnr)  # Average over z-slices
    
    
    # Loading checkpoint
    checkpoint_path = ""
    
    if os.path.exists(checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {checkpoint_path}\033[0m")
        recon_enf_params, recon_enf_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
    
    
    # Training loop for fitting the ENF backbone
    num_samples = len(train_dloader) * config.train.batch_size
    logging.info(f"Initializing latens for {num_samples} samples")
       
    # Initialize latents for the entire dataset (for each volume, all slices) (subset of LVEF can be used)
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
    for epoch in range(config.train.num_epochs_train):

        epoch_loss = []
        
        # time epoch
        start_time = time.time()

        for i, (img, patient_ids) in enumerate(train_dloader):  # Unpack the tuple
            # Convert img to JAX array
            img = jnp.array(img)
            
            # Reshape to include z dimension
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Subsample points for training
            key, subkey = jax.random.split(key)
            x_i, y_i = subsample_xy(subkey, x, y)
            
            # Get latents for this batch
            z = jax.tree.map(
                lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], 
                z_dataset
            )
        
            (loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                x_i, y_i, z, recon_enf_params, recon_enf_opt_state, key)
            
            # Update dataset with new latents 
            z_dataset = jax.tree_map(
                lambda full, partial: full.at[i*config.train.batch_size:(i+1)*config.train.batch_size].set(partial), 
                z_dataset, 
                z
            )

            epoch_loss.append(loss)
            glob_step += 1
            
        end_time = time.time()
        logging.info(f"Epoch {epoch} took {end_time - start_time} seconds")
            
        # Evaluate on first 50 samples of train set 
        train_dloader_subset = itertools.islice(train_dloader, 2)
        psnrs_evaluation = []
        
        
        if epoch % 10 == 0:
            # time evaluation
            start_time = time.time()
            
            for i, (img, patient_ids) in enumerate(train_dloader_subset):  # Unpack here too
                img = jnp.array(img)  # Convert to JAX array
                z = jax.tree_map(
                    lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], 
                    z_dataset
                )

                psnr_volume = []

                for t in range(len(config.dataset.t_indices)): 
                    x_i = x[:, t*num_subsampled_points:(t+1)*num_subsampled_points,:] # sample per volume
                    img_i = img[:,t,:,:,:,:]

                    psnr = evaluate_batch(recon_enf_params, x_i, img_i, z)

                    psnr_volume.append(psnr)

                psnrs_evaluation.append(sum(psnr_volume) / len(psnr_volume))
                
            end_time = time.time()
            logging.info(f"Evaluation took {end_time - start_time} seconds")
            
            if sum(psnrs_evaluation) / len(psnrs_evaluation) > best_psnr:
                best_psnr = sum(psnrs_evaluation) / len(psnrs_evaluation)
                logging.info(f"New best PSNR: {best_psnr}")
                save_checkpoint(f"model_checkpoints_autodecoding/{config.exp_name}/{config.run_name}", recon_enf_params, recon_enf_opt_state, epoch, glob_step, best_psnr)
                
            # Also visualise reconstruction on first sample of the train set 
            img, patient_ids = next(iter(train_dloader))  # Unpack here
            img = jnp.array(img)  # Convert to JAX array
            z = jax.tree_map(
                lambda x: x[:config.train.batch_size], 
                z_dataset
            )

            key = jax.random.PRNGKey(int(time.time()))  # Use current time as seed
            t = jax.random.randint(key, shape=(), minval=0, maxval=len(config.dataset.t_indices))

            x_i = x[:, t*num_subsampled_points:(t+1)*num_subsampled_points,:]
            img_i = img[:,t,:,:,:,:] 

            img_i_r = recon_enf.apply(recon_enf_params, x_i, *z).reshape(img_i.shape) # ONLY CONSTRUCTS ONE T VOLUME 

            fig = plot_biobank_comparison(
                img_i[0], 
                img_i_r[0], 
                t,  # Pass the time index
                config.dataset.t_indices  # Pass the full list of time indices
            )
            
            logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)} || psnr: {sum(psnrs_evaluation) / len(psnrs_evaluation)}")
            wandb.log({"train-mse": sum(epoch_loss) / len(epoch_loss), "eval-psnr": sum(psnrs_evaluation) / len(psnrs_evaluation), "reconstruction": fig, "epoch": epoch}, step=glob_step)
            plt.close('all')
   
    run.finish()


if __name__ == "__main__":
    app.run(main)
