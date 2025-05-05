import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging
from tqdm import tqdm
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
    train_dloader, test_dloader = get_dataloaders('2d_biobank', config.train.batch_size, config.dataset.num_workers, num_train=config.dataset.num_patients_train, num_test=config.dataset.num_patients_test, seed=config.seed, shuffle_train=False)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:] # Image shape : (128, 160) : 20 480 datapoints

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
                
        # Sample new key
        return (loss, z), enf_params, enf_opt_state, subkey

 
    # Training loop for fitting the ENF backbone

    num_samples = len(train_dloader) * config.train.batch_size
    logging.info(f"Initializing latens for {num_samples} samples")
       
    glob_step = 0
    z_dataset = []

    for epoch in range(config.train.num_epochs_train):

        epoch_loss = []

        for i, (img, _) in tqdm(enumerate(train_dloader), total=len(train_dloader), desc=f"Epoch {epoch}"):
            
            if epoch==0: 
                z = initialize_latents(
                    batch_size=config.train.batch_size,
                    num_latents=config.recon_enf.num_latents,
                    latent_dim=config.recon_enf.latent_dim,
                    data_dim=config.recon_enf.num_in,
                    bi_invariant_cls=TranslationBI,
                    key=key,  
                )
                
                y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
        
                (loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                    x, y, z, recon_enf_params, recon_enf_opt_state, key)
                
                z_dataset.append(z)
            else: 
                z = z_dataset[i]
                
                y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

                # Train step 
                (loss, z), recon_enf_params, recon_enf_opt_state, key = train_step(
                    x, y, z, recon_enf_params, recon_enf_opt_state, key)
                
                z_dataset[i] = z
                
            epoch_loss.append(loss)
            glob_step += 1
                
        img, _ = next(iter(train_dloader))
        z = z_dataset[0]

        img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
        fig = plot_biobank_comparison(img[0], img_r[0], poses=z[0][0])
        
        logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)}")
        wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "reconstruction": fig}, step=glob_step)
        plt.close('all')

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
