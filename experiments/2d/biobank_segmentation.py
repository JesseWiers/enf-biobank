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

jax.config.update("jax_default_matmul_precision", "highest")

def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
    mask: jnp.ndarray = None, 
    pred_mask: jnp.ndarray = None, 
    save_path: str = None
):
    """Plot original and reconstructed organ images side by side.
    
    Args:
        original: Original images with shape (H, W, 1)
        reconstruction: Reconstructed images with shape (H, W, 1)
        mask: Optional ground truth organs mask with shape (H, W)
        pred_mask: Optional predicted organs mask with shape (H, W)
        save_path: Optional path to save the figure
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Original (top) vs Reconstruction (bottom)')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1) # 

    # Plot originals
    
    # Plot original
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    
    
    # Plot reconstruction
    axes[1,0].imshow(reconstruction, cmap='gray')
    axes[1,0].set_title('Reconstructed Image')
    
    
    # Plot poses
    if poses is not None:
        # Map to 0-W range
        poses = (poses + 1) * original.shape[0] / 2
        axes[0,1].scatter(poses[:, 0], poses[:, 1], c='r', s=10)
        axes[0,1].set_title('Pose')
    else:
        axes[0,1].axis('off')
        
        
    # TODO: Pred_masks area already grayscale but mask is not yet
    
    # Plot masks
    if mask is not None:
        axes[0,2].imshow(mask, cmap='gray')
        axes[0,2].set_title('Ground Truth Mask')
    else:
        axes[0,2].axis('off')
        
    if pred_mask is not None:
        axes[1,2].imshow(pred_mask, cmap='gray')
        axes[1,2].set_title('Predicted Mask')
    else:
        axes[1,2].axis('off')
        
    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig
    

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
    config.recon_enf.freq_mult = (3.0, 5.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 16
    config.recon_enf.latent_dim = 64
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True
    
    # Segmentation model
    config.seg_enf = ml_collections.ConfigDict()
    config.seg_enf.num_hidden = 128 # Original = 128
    config.seg_enf.num_heads = 3 # Original = 3
    config.seg_enf.att_dim = 128 # Original = 128
    config.seg_enf.num_in = 2 # Original = 2
    config.seg_enf.num_out = 4 # logits of 4 classes  
    config.seg_enf.freq_mult = (2.0, 10.0) # Original = (2.0, 10.0)
    config.seg_enf.k_nearest = 4 # Original = 4

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
    config.train.num_epochs_train_seg = 10
    config.train.log_interval = 50
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())



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
    
    seg_enf = EquivariantNeuralField(
        num_hidden=config.seg_enf.num_hidden,
        att_dim=config.seg_enf.att_dim,
        num_heads=config.seg_enf.num_heads,
        num_out=config.seg_enf.num_out,
        emb_freq=config.seg_enf.freq_mult,
        nearest_k=config.seg_enf.k_nearest,
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
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)
    seg_enf_params = seg_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)
    seg_enf_opt_state = enf_opt.init(seg_enf_params)

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
    
    
    def evaluate_test_set(enf_params, test_dloader, key):
        """Evaluate model on the entire test set."""
        psnrs = []
        mses = []
        reconstructions = []
        originals = []
        
        for img, _ in test_dloader:
            
            # Flatten input
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Get new key
            key, subkey = jax.random.split(key)
            
            mse, (z, psnr_value) = recon_inner_loop(enf_params, x, y, key)
            
            psnrs.append(psnr_value)
            mses.append(mse)
            img_r = recon_enf.apply(enf_params, x, *z).reshape(img.shape)
            
            originals.append(img)
            reconstructions.append(img_r)
                 
        # Calculate average PSNR
        avg_psnr = jnp.mean(jnp.array(psnrs))
        avg_mse = jnp.mean(jnp.array(mses))
        
        # Return first batch for visualization
        return avg_psnr, avg_mse, originals[0], reconstructions[0]
    
    
    ##### SEGMENTATION CODE #####
    
    @jax.jit
    def segment_outer_step(coords, img, gt_mask, recon_enf_params, seg_enf_params, seg_enf_opt_state, key):        
        # Perform inner loop optimization to obtain latent
        key, subkey = jax.random.split(key)
        loss, (z, psnr_value) = recon_inner_loop(recon_enf_params, coords, img, key)
                
        # Calculate the segmentation loss
        def CE_loss(seg_enf_params, z, coords, gt_mask):
            # Get logits from the segmentation model
            logits = seg_enf.apply(seg_enf_params, coords, *z)  # (B, H, W, 4)
            
            # One-hot encode the class indices 
            one_hot_labels = jax.nn.one_hot(gt_mask, num_classes=4) # (B, H, W, 1) -> (B, H, W, Classes)
            
            # Ensure one_hot_labels has the same shape as logits
            one_hot_labels = one_hot_labels.reshape(logits.shape)
            
            # Compute cross-entropy loss
            loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            
            # Return the mean loss over all pixels and batch elements
            return jnp.mean(loss)
            
        # Update the segmentation ENF
        loss, grads = jax.value_and_grad(CE_loss)(seg_enf_params, z, coords, gt_mask)
        
        enf_grads, seg_enf_opt_state = enf_opt.update(grads, seg_enf_opt_state)
        seg_enf_params = optax.apply_updates(seg_enf_params, enf_grads)

        # Sample new key
        return (loss, z), seg_enf_params, seg_enf_opt_state, subkey
    
    #################################
    
    


    checkpoint_path = "/home/jwiers/deeprisk/new_codebase/enf-biobank/checkpoints/recon_phase/checkpoint_4600.pkl"
    
    if os.path.exists(checkpoint_path):
        print(f"\033[93mResuming training from checkpoint: {checkpoint_path}\033[0m")
        recon_enf_params, recon_enf_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        print("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
        
        
    
    
 
    # Pretraining loop for fitting the ENF backbone
    best_psnr = float('-inf')
    glob_step = 0
    
    for epoch in range(config.train.num_epochs_train_seg):
        epoch_loss = []
        for i, (img, mask) in enumerate(train_dloader):
            
            # flatten input
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            gt_mask = jnp.reshape(mask, (mask.shape[0], -1, mask.shape[-1]))

            # Perform outer loop optimization
            (loss, z), seg_enf_params, seg_enf_opt_state, key = segment_outer_step(
                x, y, gt_mask, recon_enf_params, seg_enf_params, seg_enf_opt_state, key)
            
            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                
                
                test_iter = iter(test_dloader)
                random_sample_idx = jax.random.randint(key, (), 0, len(test_dloader))
                for i in range(random_sample_idx):
                    img, mask = next(test_iter)
    
                # img, mask = next(iter(test_dloader))
                
                
                y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
                
                _, (z, _) = recon_inner_loop(recon_enf_params, x, y, key)
                
                img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
                
                logits = seg_enf.apply(seg_enf_params, x, *z)
                logits = logits.reshape(img.shape[0], img.shape[1], img.shape[2], 4) # (B, H, W, 4)
                probabilities = jax.nn.softmax(logits, axis=-1)
                predicted_classes = jnp.argmax(probabilities, axis=-1)
                class_to_grayscale = jnp.array([0, 84, 171, 255])
                
                pred_mask = class_to_grayscale[predicted_classes]
                mask = class_to_grayscale[mask.astype(jnp.int32)]
                
                fig = plot_biobank_comparison(img[0], img_r[0], mask=mask[0], pred_mask=pred_mask[0])
                
                wandb.log({"seg-loss": sum(epoch_loss) / len(epoch_loss), "seg-reconstruction": fig}, step=glob_step)
                logging.info(f"SEG ep {epoch} / step {glob_step} || loss: {sum(epoch_loss) / len(epoch_loss)}")
                
                # test_psnr, test_mse, img, img_r = evaluate_test_set(recon_enf_params, test_dloader, key)
                
                # fig = plot_biobank_comparison(img[0], img_r[0], poses=z[0][0])
                
                # wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "test-mse": test_mse, "test-psnr": test_psnr, "reconstruction": fig}, step=glob_step)
                # plt.close('all')
                # logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])} || test-mse: {test_mse} || test-psnr: {test_psnr}")

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
