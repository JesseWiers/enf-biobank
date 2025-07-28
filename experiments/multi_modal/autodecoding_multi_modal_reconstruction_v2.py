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
import itertools
# Custom imports
from experiments.datasets.biobank_multimodal_endpoint_dataset import get_dataloaders
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
    config.recon_enf.num_out = 13  
    config.recon_enf.freq_mult = (30.0, 60.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 1296 
    config.recon_enf.latent_dim = 32
    config.recon_enf.z_positions = 2
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True
    
    config.recon_enf.num_subsamples_multiplier = 100
    
    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients = 100
    config.dataset.z_indices = (0, 1, 2, 3, 4, 5, 6, 7)
    config.dataset.t_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4 
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window), orginally (2., 30., 0.) # NOTE: Try 1e-3 
    
    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 1
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 1000
    config.train.log_interval = 50
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

def plot_ecg_reconstructions(original_ecg, reconstructed_ecg, epoch):
    """
    Create visualization comparing original and reconstructed ECG leads.
    
    Args:
        original_ecg: Original ECG data of shape [12, T]
        reconstructed_ecg: Reconstructed ECG data of shape [12, T]
        epoch: Current epoch number for title
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle(f'ECG Lead Reconstructions - Epoch {epoch}')
    
    for lead in range(12):
        row = lead // 3
        col = lead % 3
        
        # Plot original
        axes[row, col].plot(original_ecg[lead], 'b-', label='Original', alpha=0.6)
        # Plot reconstruction
        axes[row, col].plot(reconstructed_ecg[lead], 'r--', label='Reconstructed', alpha=0.6)
        
        axes[row, col].set_title(f'Lead {lead+1}')
        axes[row, col].grid(True)
        if lead == 0:  # Only add legend to first plot
            axes[row, col].legend()
    
    plt.tight_layout()
    return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader = get_dataloaders(
        dataset_name='multi_modal',
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        num_patients=config.dataset.num_patients,  # Changed from num_patients_train
        seed=config.seed,
        z_indices=config.dataset.z_indices,
        t_indices=config.dataset.t_indices
    )
    sample_img, sample_ecg, _ = next(iter(train_dloader))
    img_shape = sample_img[0].shape # [T, Z, H, W]


    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x_mri = create_coordinate_grid(
        batch_size=config.train.batch_size, 
        img_shape=img_shape,
        num_in=config.recon_enf.num_in  # 4D coordinates
    ) # [T * Z * H * W]

    ecg_shape = (sample_ecg.shape[2], 8, 71, 77)

    x_ecg = create_coordinate_grid(
        batch_size=config.train.batch_size, 
        img_shape=ecg_shape,
        num_in=config.recon_enf.num_in  # 4D coordinates
    ) # [T * Z * H * W] -> Each Timepoint has Z * H * W Possible indices to sample 

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

    num_subsampled_points = 300 * config.recon_enf.num_subsamples_multiplier

    # Init the model
    recon_enf_params = recon_enf.init(key, x_mri[:, :num_subsampled_points, :], *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)


    @jax.jit
    def subsample_xy_mri(key, x_, y_):
        """Subsample coordinates and pixel values.
        """
        num_points = x_.shape[1]
        sub_mask = jax.random.permutation(key, num_points)[:num_subsampled_points]
        x_i = x_[:, sub_mask]
        y_i = y_[:, sub_mask]
        return x_i, y_i


    @jax.jit
    def subsample_xy_ecg(key, x_, y_):
        # x_ shape: (B, T*Z*H*W, 4)
        # y_ shape: (B, 12, T, 1)
        
        # Reshape x to (B, T, Z*H*W, 4)
        x_reshaped = x_.reshape(x_.shape[0], 300, -1, 4)  # Shape: (B, T, Z*H*W, 4)
        
        points_per_t = num_subsampled_points // 300
        
        # Create subkeys for each timepoint
        keys = jax.random.split(key, 300)
        
        # Sample indices for each timepoint at once
        indices = jax.vmap(lambda k: jax.random.choice(
            k, 
            8*71*77, 
            shape=(points_per_t,), 
            replace=False
        ))(keys)  # Shape: (T, points_per_t)
        
        # Add batch dimension to indices for broadcasting
        indices = indices[None, :, :, None]  # Shape: (1, T, points_per_t, 1)
        
        # Gather points using these indices, maintaining batch and coordinate dimensions
        sampled_x = jnp.take_along_axis(x_reshaped, indices, axis=2)  # Shape: (B, T, points_per_t, 4)
        
        # For y_, we need to repeat each timepoint's values points_per_t times
        y_expanded = jnp.repeat(y_[..., None], points_per_t, axis=3)
        
        # Reshape both outputs to final shapes
        sampled_x_final = sampled_x.reshape(x_.shape[0], -1, 4)  # Shape: (B, T*points_per_t, 4)
        y_final = y_expanded.reshape(y_.shape[0], 12, -1, 1)  # Shape: (B, 12, T*points_per_t, 1)
        
        return sampled_x_final, y_final

    @jax.jit
    def train_step(x_i_mri, x_i_ecg, y_i_total, z, enf_params, enf_opt_state, key):
        def loss_fn(z, enf_params):
            # Forward pass for both coordinate sets
            y_i_hat_mri = recon_enf.apply(enf_params, x_i_mri, *z)  # (B, num_points, 13)
            y_i_hat_ecg = recon_enf.apply(enf_params, x_i_ecg, *z)  # (B, num_points, 13)
            
            # Compute MRI loss (only channel 0)
            mri_loss = jnp.mean((y_i_hat_mri[..., 0] - y_i_total[..., 0]) ** 2)
            
            # Compute ECG loss (channels 1-12)
            ecg_loss = jnp.mean((y_i_hat_ecg[..., 1:] - y_i_total[..., 1:]) ** 2)
            
            # Total loss
            total_loss = mri_loss + ecg_loss
            return total_loss, (mri_loss, ecg_loss)

        key, subkey = jax.random.split(key)
        (loss, (mri_loss, ecg_loss)), (z_grads, enf_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(z, enf_params)
        
        # Update latents and model parameters
        z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, z_grads, config.optim.inner_lr)
        enf_grads, enf_opt_state = enf_opt.update(enf_grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)
        
        return (loss, mri_loss, ecg_loss, z), enf_params, enf_opt_state, subkey
    
    
    @jax.jit
    def evaluate_mri_batch(enf_params, x_i_mri, mri_i, z):
        
        # Compute reconstructions
        mri_i_hat = recon_enf.apply(enf_params, x_i_mri, *z)[..., 0].reshape(mri_i.shape)
        
        # Compute PSNR per slice and average
        mse = jnp.mean((mri_i - mri_i_hat) ** 2, axis=(1, 2, 4))  # Average over H, W, channels
        max_pixel_value = 1.0
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
        return jnp.mean(psnr)  # Average over z-slices
    
    
    @jax.jit
    def evaluate_ecg_batch(enf_params, x_i_ecg, ecg_i, z):
        # Compute reconstructions
        ecg_i_hat = recon_enf.apply(enf_params, x_i_ecg, *z)[..., 1:]  # Shape: (B, 300, 12)
        
        # Reshape ecg_i to match ecg_i_hat
        ecg_i = jnp.transpose(ecg_i, (0, 2, 1, 3))[..., 0]  # Shape: (B, 300, 12)
        
        # Compute MSE per lead
        mse = jnp.mean((ecg_i - ecg_i_hat) ** 2, axis=1)  # Average over timepoints, shape: (B, 12)
        
        # Compute PSNR per lead
        max_pixel_value = 1.0
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))  # Shape: (B, 12)
        
        # Average PSNR across leads
        return jnp.mean(psnr)  # Average over leads to get one value per batch
        
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
    
    # obtain x_ecg_lead 
    stride = 8 * 71 * 77  # Z*H*W
    indices = jnp.arange(0, 300 * stride, stride)  # Will give us 300 indices, one for each timepoint
    x_ecg_lead = x_ecg[:, indices, :]  # Shape will be [B, 300, 4]
    # TODO: WHICH X,Y,Z COORDINATES ARE USED FOR THE ECG
    
    pixels_volume = 71 * 77 * 8

    for epoch in range(config.train.num_epochs_train):

        epoch_loss = []
        epoch_mri_loss = []
        epoch_ecg_loss = []
        
        start_time = time.time()
        for i, (mri, ecg, _) in enumerate(train_dloader):
            
            # Reshape to include z dimension
            y_mri = jnp.reshape(mri, (mri.shape[0], -1, mri.shape[-1]))
            y_ecg = ecg # Shape -> (1, 21, 300, 1)
            
            # Subsample points for training
            key, subkey = jax.random.split(key)
            
            x_i_mri, y_i_mri = subsample_xy_mri(subkey, x_mri, y_mri) # y_i_mri: (1, num_subsampled_points, 1)
            x_i_ecg, y_i_ecg = subsample_xy_ecg(subkey, x_ecg, y_ecg) # y_i_ecg: (1, 12, num_subsampled_points, 1)
            
            # Get y_i_total
            y_i_ecg_transposed = jnp.transpose(y_i_ecg, (0, 2, 1, 3))
            y_i_mri = y_i_mri[..., 0]  # (1, 3000)
            y_i_ecg_transposed = y_i_ecg_transposed[..., 0]  # (1, 3000, 12)
            y_i_total = jnp.concatenate([y_i_mri[..., None], y_i_ecg_transposed], axis=-1)  # (1, num_subsampled_points, 13)

        
            # Get latents for this batch
            z = jax.tree.map(
                lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], 
                z_dataset
            )

            # Get combined and individual losses    
            (loss, mri_loss, ecg_loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                x_i_mri, x_i_ecg, y_i_total, z, recon_enf_params, recon_enf_opt_state, key
            )
            
            # Update dataset with new latents 
            z_dataset = jax.tree_map(
                lambda full, partial: full.at[i*config.train.batch_size:(i+1)*config.train.batch_size].set(partial), 
                z_dataset, 
                z
            )

            epoch_loss.append(loss)
            epoch_mri_loss.append(mri_loss)
            epoch_ecg_loss.append(ecg_loss)
            glob_step += 1
            
        end_time = time.time()
        logging.info(f"Epoch {epoch} took {end_time - start_time} seconds")
        
        # Evaluation 
        if epoch % 2 == 0:

            # time evaluation
            start_time = time.time()
        
            #### PSNR Computation  ####
            
            psnrs_evaluation_mris = []
            psnrs_evaluation_ecgs = []
            
            n_eval_samples = 10 
            train_dloader_subset = itertools.islice(train_dloader, n_eval_samples)
            
            for i, (mri, ecg, _) in enumerate(tqdm(train_dloader_subset, desc=f"Performing evaluation, epoch: {epoch}", total=n_eval_samples)):
                z = jax.tree_map(
                    lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], 
                    z_dataset
                )
                
                # Compute PSNR for entire MRI volume 
                psnr_mri_volume = []
                
                for t in range(len(config.dataset.t_indices)):
                    
                    x_i_mri_volume = x_mri[:, t*pixels_volume:(t+1)*pixels_volume,:]
                    mri_i_volume = mri[:,t,:,:,:,:]
                    
                    psnr_volume = evaluate_mri_batch(recon_enf_params, x_i_mri_volume, mri_i_volume, z)
                    psnr_mri_volume.append(psnr_volume)
                    
                psnrs_evaluation_mris.append(sum(psnr_mri_volume) / len(psnr_mri_volume))
                
                # Compute PSNR for ecg 
                
                # x_ecg_lead -> [B, 300, 4]
                # ecg -> [B, 300, 12, 1]
                
                psnr_ecg = evaluate_ecg_batch(recon_enf_params, x_ecg_lead, ecg, z)
                psnrs_evaluation_ecgs.append(psnr_ecg)
                      
            # logging metrics
            wandb.log({
                "eval_psnr_mris": jnp.mean(jnp.array(psnrs_evaluation_mris)),
                "eval_psnr_ecgs": jnp.mean(jnp.array(psnrs_evaluation_ecgs)),
                "epoch_loss": jnp.mean(jnp.array(epoch_loss)),
                "epoch_mri_loss": jnp.mean(jnp.array(epoch_mri_loss)),
                "epoch_ecg_loss": jnp.mean(jnp.array(epoch_ecg_loss)),
            }, step=epoch)
            
            logging.info(f"Epoch {epoch} Loss: {jnp.mean(jnp.array(epoch_loss))} MRI Loss: {jnp.mean(jnp.array(epoch_mri_loss))} ECG Loss: {jnp.mean(jnp.array(epoch_ecg_loss))}")
            logging.info(f"Epoch {epoch} PSNR MRI: {jnp.mean(jnp.array(psnrs_evaluation_mris))} PSNR ECG: {jnp.mean(jnp.array(psnrs_evaluation_ecgs))}")
                             
            #### Visualisation ####
            
            # Generate random batch index
            batch_idx = jax.random.randint(
                key, 
                shape=(), 
                minval=0, 
                maxval=len(train_dloader)
            )
            
            # Get that specific batch
            for i, (mri, ecg, _) in enumerate(train_dloader):
                if i == batch_idx:
                    
                    # Obtain z corresponding to this batch
                    z = jax.tree_map(
                        lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], 
                        z_dataset
                    )
                    
                    # Visualize ECG Leads 
                    y_i_hat_ecg_lead = recon_enf.apply(recon_enf_params, x_ecg_lead, *z)[..., 1:]
                    
                    # Prepare data for plotting
                    original_ecg = ecg[0, :, :, 0]  # Shape: [12, 300]
                    reconstructed_ecg = y_i_hat_ecg_lead[0].T  # Shape: [12, 300]
                    
                    # Create and log visualization
                    fig = plot_ecg_reconstructions(original_ecg, reconstructed_ecg, epoch)
                    wandb.log({
                        "ecg_reconstructions": wandb.Image(fig),
                        "epoch": epoch,
                        "sample_idx": batch_idx
                    }, step=glob_step)
                    plt.close(fig)
                    
                    # Visualize MRI Reconstructions 
                    key = jax.random.PRNGKey(int(time.time()))
                    t = jax.random.randint(key, shape=(), minval=0, maxval=len(config.dataset.t_indices))

                    x_mri_i_volume = x_mri[:, t*pixels_volume:(t+1)*pixels_volume,:]
                    mri_i_volume = mri[:,t,:,:,:,:] 

                    mri_i_volume_r = recon_enf.apply(recon_enf_params, x_mri_i_volume, *z)[..., 0].reshape(mri_i_volume.shape)

                    fig = plot_biobank_comparison(
                        mri_i_volume[0], 
                        mri_i_volume_r[0], 
                        t,
                        config.dataset.t_indices
                    )
                    
                    wandb.log({
                        "mri_reconstructions": wandb.Image(fig),
                        "sample_idx": batch_idx,
                        "timepoint": t
                    }, step=epoch)
                    plt.close(fig)
                    
                    break
                
            end_time = time.time()
            logging.info(f"Evaluation took {end_time - start_time} seconds")

    run.finish()

if __name__ == "__main__":
    app.run(main)
