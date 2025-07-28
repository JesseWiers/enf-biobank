import ml_collections
from ml_collections import config_flags
from absl import app

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import traceback
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

# Custom imports
from ResNet import ResNet50
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
import logging
import time

class CardiacMRIDataset(Dataset):
    def __init__(self, nifti_root_path, endpoints_csv_path, latent_hdf5_path, endpoint_name='LVEF', 
                 use_rotated=False, debug_limit=None):
        """
        Dataset for loading cardiac MRI images directly from Nifti files.
        
        Args:
            nifti_root_path (str): Path to directory containing patient Nifti files
            endpoints_csv_path (str): Path to CSV with endpoints
            latent_hdf5_path (str): Path to HDF5 file containing rotation information
            endpoint_name (str): Name of the endpoint column
            use_rotated (bool): Whether to use rotated versions
            debug_limit (int, optional): Limit number of patients for debugging
        """
        self.nifti_root_path = nifti_root_path
        self.endpoint_name = endpoint_name
        self.use_rotated = use_rotated
        self.latent_hdf5_path = latent_hdf5_path
        
        # Load endpoints CSV
        self.endpoints_df = pd.read_csv(endpoints_csv_path)
        
        # Get patient ID column
        self.patient_id_col = next((col for col in self.endpoints_df.columns 
                                  if col.lower() in ['f.eid']), None)
        if self.patient_id_col is None:
            raise ValueError("Could not find patient ID column in endpoints CSV")
        
        self.endpoints_df[self.patient_id_col] = self.endpoints_df[self.patient_id_col].astype(str)
        
        # Filter patients without endpoint or ED/ES timepoints
        self.endpoints_df = self.endpoints_df.dropna(subset=[endpoint_name, 'ED', 'ES'])
        
        if debug_limit:
            self.endpoints_df = self.endpoints_df.head(debug_limit)
            
        self.patient_ids = self.endpoints_df[self.patient_id_col].tolist()
        
        # Load rotation information from HDF5 file
        self.rotations = {}
        with h5py.File(latent_hdf5_path, 'r') as f:
            for patient_id in self.patient_ids:
                if f'patient_{patient_id}' in f:
                    patient_group = f[f'patient_{patient_id}']
                    if 'rotation' in patient_group.attrs:
                        self.rotations[patient_id] = patient_group.attrs['rotation']
        
        logging.info(f"Dataset contains {len(self.patient_ids)} patients")
        
    def rotate_image(self, image, k):
        """Rotate image by k * 90 degrees"""
        return np.rot90(image, k=k)
    
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            
            # Get endpoint value and ED/ES timepoints
            patient_data = self.endpoints_df.loc[
                self.endpoints_df[self.patient_id_col] == patient_id
            ]
            endpoint_value = patient_data[self.endpoint_name].values[0]
            ed_timepoint = int(patient_data['ED'].values[0])
            es_timepoint = int(patient_data['ES'].values[0])
            
            # Load Nifti file
            nifti_path = f"{self.nifti_root_path}/{patient_id}/cropped_sa.nii.gz"
            nifti_img = nib.load(nifti_path)
            image_data = nifti_img.get_fdata()  # Shape: [H, W, Z, T]
            
            # Get middle slice
            Z = image_data.shape[2]
            middle_z = Z // 2
            
            # Extract ED and ES frames from middle slice
            ed_frame = image_data[:, :, middle_z, ed_timepoint]
            es_frame = image_data[:, :, middle_z, es_timepoint]
            
            # Stack ED and ES frames as channels
            combined_image = np.stack([ed_frame, es_frame], axis=0)
            
            # Normalize each frame independently
            for i in range(combined_image.shape[0]):
                frame = combined_image[i]
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                combined_image[i] = (frame - frame_min) / (frame_max - frame_min)
            
            # Apply rotation if specified and rotation exists for this patient
            if self.use_rotated and patient_id in self.rotations:
                k = self.rotations[patient_id] // 90  # Convert degrees to number of 90° rotations
                combined_image = self.rotate_image(combined_image, k)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(combined_image).float()
            
            # Create binary label (0 if LVEF < 40, 1 if LVEF >= 40)
            label = 1 if endpoint_value >= 40.0 else 0
            
            return patient_id, image_tensor, label
            
        except Exception as e:
            logging.error(f"Error loading patient {patient_id}: {str(e)}")
            # Return first item as fallback
            if idx > 0:
                return self.__getitem__(0)
            else:
                # Create empty placeholder
                return "error", torch.zeros((2, 128, 128)), 0

def create_split_indices(dataset_size, train_split=0.7, val_split=0.15, random_seed=42):
    """
    Create deterministic train/val/test split indices.
    
    Args:
        dataset_size (int): Total number of samples
        train_split (float): Proportion for training (default: 0.7)
        val_split (float): Proportion for validation (default: 0.15)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    np.random.seed(random_seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def create_dataloaders(nifti_root_path, endpoints_csv_path, latent_hdf5_path, batch_size=16, 
                      train_split=0.7, val_split=0.15, random_seed=42, 
                      num_workers=0, debug_limit=None):
    """Create train, validation and test dataloaders."""
    
    # Create datasets
    train_val_dataset = CardiacMRIDataset(
        nifti_root_path, 
        endpoints_csv_path,
        latent_hdf5_path,
        use_rotated=False,
        debug_limit=debug_limit
    )
    
    test_dataset = CardiacMRIDataset(
        nifti_root_path,
        endpoints_csv_path,
        latent_hdf5_path,
        use_rotated=True,
        debug_limit=debug_limit
    )
    
    # Get dataset size and create split indices
    dataset_size = len(train_val_dataset)
    train_indices, val_indices, test_indices = create_split_indices(
        dataset_size, train_split, val_split, random_seed
    )
    
    # Create dataloaders with the same indices
    train_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_config():
    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 4
    config.dataset.nifti_root = "/projects/prjs1252/data_jesse/cmr_cropped"
    config.dataset.csv_path = "/projects/prjs1252/data_jesse/metadata/filtered_endpoints.csv"
    config.dataset.latent_hdf5_path = "/projects/prjs1252/data_jesse/latent_dataset_4d.h5"
    config.dataset.debug_limit = None

    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "ResNet50"
    config.model.num_classes = 2
    config.model.in_channels = 2  # ED and ES frames

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 32
    config.train.num_epochs = 100
    config.train.validation_interval = 10
    config.train.learning_rate = 0.001

    # Set checkpoint path
    config.exp_name = "lvef_prediction_resnet"
    config.run_name = "baseline"
    
    logging.getLogger().setLevel(logging.INFO)
    
    return config

# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())

def main(_):
    # Get config
    config = _CONFIG.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online", name=config.run_name)

    # Load dataset with new splits
    train_loader, val_loader, test_loader = create_dataloaders(
        nifti_root_path=config.dataset.nifti_root,
        endpoints_csv_path=config.dataset.csv_path,
        latent_hdf5_path=config.dataset.latent_hdf5_path,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit,
        train_split=0.7,
        val_split=0.15
    )
    
    logging.info(f"Train dataloader length: {len(train_loader)}")
    logging.info(f"Val dataloader length: {len(val_loader)}")
    logging.info(f"Test dataloader length: {len(test_loader)}")

    # Create model
    model = ResNet50(
        num_classes=config.model.num_classes,
        channels=config.model.in_channels
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)

    # Variables to track best model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    # Training loop
    global_step = 0
    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_losses = []
        
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            patient_ids, images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_step += 1

        # Validation phase at end of each epoch
        model.eval()
        avg_loss = np.mean(epoch_losses)
        wandb.log({"train/loss": avg_loss, "epoch": epoch}, step=global_step)
        
        val_losses = []
        val_accuracies = []
        misclassified_data = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation epoch {epoch}"):
                patient_ids, images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                
                accuracy = (preds == labels).float().mean().item()
                
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)
                
                # Track misclassified patients
                misclassified_mask = preds != labels
                if misclassified_mask.any():
                    misclassified_indices = torch.where(misclassified_mask)[0]
                    for idx in misclassified_indices:
                        misclassified_data.append({
                            'patient_id': patient_ids[idx],
                            'true_lvef': "LVEF ≥ 40" if labels[idx].item() == 1 else "LVEF < 40",
                            'predicted_class': "LVEF ≥ 40" if preds[idx].item() == 1 else "LVEF < 40"
                        })
        
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        
        # Track best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_model_state = model.state_dict().copy()  # Make sure to copy the state dict
            best_epoch = epoch
            
            logging.info(f"New best model at epoch {epoch} with accuracy {best_val_accuracy:.4f}")
        
        wandb.log({
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_accuracy,
            "val/best_accuracy": best_val_accuracy,
            "epoch": epoch
        }, step=global_step)
        
        # Log misclassified patients
        if misclassified_data:
            misclassified_table = wandb.Table(
                columns=["patient_id", "true_lvef", "predicted_class"]
            )
            for item in misclassified_data:
                misclassified_table.add_data(
                    item['patient_id'],
                    item['true_lvef'],
                    item['predicted_class']
                )
            wandb.log({f"val/misclassified_patients_epoch_{epoch}": misclassified_table})
        
        model.train()

    # Final test set evaluation using rotated versions
    logging.info(f"\nEvaluating on test set using best model from epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_losses = []
    test_accuracies = []
    misclassified_data = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test set evaluation"):
            patient_ids, images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            
            accuracy = (preds == labels).float().mean().item()
            
            test_losses.append(loss.item())
            test_accuracies.append(accuracy)
            
            # Track misclassifications
            misclassified_mask = preds != labels
            if misclassified_mask.any():
                misclassified_indices = torch.where(misclassified_mask)[0]
                for idx in misclassified_indices:
                    misclassified_data.append({
                        'patient_id': patient_ids[idx],
                        'true_lvef': "LVEF ≥ 40" if labels[idx].item() == 1 else "LVEF < 40",
                        'predicted_class': "LVEF ≥ 40" if preds[idx].item() == 1 else "LVEF < 40"
                    })

    # Log final test results
    final_test_loss = np.mean(test_losses)
    final_test_accuracy = np.mean(test_accuracies)
    
    logging.info(f"\nTest Set Results (using rotated versions):")
    logging.info(f"Loss: {final_test_loss:.4f}")
    logging.info(f"Accuracy: {final_test_accuracy:.4f}")
    
    wandb.log({
        "test/loss": final_test_loss,
        "test/accuracy": final_test_accuracy,
        "test/num_misclassified": len(misclassified_data)
    })
    
    # Log misclassified cases
    if misclassified_data:
        misclassified_table = wandb.Table(
            columns=["patient_id", "true_lvef", "predicted_class"]
        )
        for item in misclassified_data:
            misclassified_table.add_data(
                item['patient_id'],
                item['true_lvef'],
                item['predicted_class']
            )
        wandb.log({"test/misclassified_patients": misclassified_table})

    run.finish()

if __name__ == "__main__":
    app.run(main)
