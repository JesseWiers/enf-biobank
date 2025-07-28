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
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 16
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 100
    config.train.log_interval = 200
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
    train_dloader, _ = get_dataloaders('biobank_lvef', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed, shuffle_train=False)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] 

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)

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
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)

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

        for i, (img, _) in enumerate(train_dloader):
            
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Extract latents corresponding to batch 
            z = jax.tree.map(lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], z_dataset)
        
            (loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                x, y, z, recon_enf_params, recon_enf_opt_state, key)
            
            # Update dataset with new latents 
            z_dataset = jax.tree.map(lambda full, partial: full.at[i*config.train.batch_size:(i+1)*config.train.batch_size].set(partial), z_dataset, z)

            epoch_loss.append(loss)
            glob_step += 1
            
        end_time = time.time()
        logging.info(f"Epoch {epoch} took {end_time - start_time} seconds")
            
        # Evaluate on first 1000 samples of train set 
        train_dloader_subset = itertools.islice(train_dloader, 1000)
        psnrs_evaluation = []
        
        # time evaluation
        start_time = time.time()
        
        for i, (img, _) in enumerate(train_dloader_subset):
            z = jax.tree.map(lambda x: x[i*config.train.batch_size:(i+1)*config.train.batch_size], z_dataset)
            psnr = evaluate_batch(recon_enf_params, x, img, z)
            psnrs_evaluation.append(psnr)
            
        end_time = time.time()
        logging.info(f"Evaluation took {end_time - start_time} seconds")
        
        if sum(psnrs_evaluation) / len(psnrs_evaluation) > best_psnr:
            best_psnr = sum(psnrs_evaluation) / len(psnrs_evaluation)
            logging.info(f"New best PSNR: {best_psnr}")
            save_checkpoint(f"model_checkpoints_autodecoding/{config.exp_name}/{config.run_name}", recon_enf_params, recon_enf_opt_state, epoch, glob_step, best_psnr)
            
        # Also visualise reconstruction on first sample of the train set 
        img, _ = next(iter(train_dloader))
        z = jax.tree.map(lambda x: x[:config.train.batch_size], z_dataset)
        img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
        fig = plot_biobank_comparison(img[0], img_r[0], poses=z[0][0])
        
        logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)} || psnr: {sum(psnrs_evaluation) / len(psnrs_evaluation)}")
        wandb.log({"train-mse": sum(epoch_loss) / len(epoch_loss), "eval-psnr": sum(psnrs_evaluation) / len(psnrs_evaluation), "reconstruction": fig, "epoch": epoch}, step=glob_step)
        plt.close('all')

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
