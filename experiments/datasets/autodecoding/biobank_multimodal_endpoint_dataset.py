import pandas as pd  # Add this import at the top
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np
import os
import jax.numpy as jnp
import time
import traceback
import nibabel as nib
from tqdm import tqdm
import torch
import torchvision
from PIL import Image
import numpy as np
import logging

from experiments.datasets.ombria_dataset import Ombria
from torchvision.datasets import CIFAR10

def image_to_numpy(image: Image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image) / 255


def numpy_collate(batch: list[np.ndarray]) -> np.ndarray:
    """
    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    num_patients: int = None,
    seed: int = 42,
    z_indices: tuple = (0, 1, 2, 3, 4, 5, 6, 7),
    t_indices: tuple = (0, 1),
    num_leads: int = 12  # Add this parameter
) -> DataLoader:
    """Get dataloaders for training.
    
    Args:
        dataset_name: Name of the dataset to use
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        num_patients: Number of patients to use (None for all)
        seed: Random seed
        z_indices: Z-indices to use
        t_indices: Time indices to use
        num_leads: Number of ECG leads to use
    
    Returns:
        DataLoader: Training dataloader
    """
    
    # Create generator with seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "multi_modal":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = EndpointDatasetMultiModal(
            root='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', 
            ecg_path='/projects/prjs1252/data_jesse_final_v3/ECGs_median_leads.pth', 
            num_patients=num_patients, 
            z_indices=z_indices, 
            t_indices=t_indices,
            num_leads=num_leads  # Pass the parameter here
        )
    else: 
        raise NotImplementedError("Dataset not implemented yet.")
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=False, # Always False as per edit hint
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,  # Add generator for reproducible shuffling
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)  # Seed workers
    )
    
    return train_dataloader

class EndpointDatasetMultiModal(Dataset):
    
    def __init__(
        self,
        root: str,  # MRI data root
        ecg_path: str,  # Path to ECG data
        num_patients: int = None,  # Number of patients to return (None for all)
        z_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        t_indices: list[int] = [0, 1],
        num_leads: int = 12,  # Add this parameter
        random_seed: int = 42,
    ):
        self.mri_data = []
        self.ecg_data = []
        self.z_indices = sorted(z_indices)
        self.t_indices = sorted(t_indices)
        self.patient_ids = []

        # Set random seed
        rng = np.random.RandomState(random_seed)
        
        # Load ECG data
        ecg_data = torch.load(ecg_path)
        self.median_heartbeats = ecg_data['median_heartbeats']
        ecg_ids = set(ecg_data['IDs'])
        
        # Get all MRI patient paths and shuffle them
        all_patient_paths = os.listdir(root)
        rng.shuffle(all_patient_paths)  # Shuffle all paths first
        
        # Find matching patients
        valid_patients = []
        for path in all_patient_paths:
            eid = path.split('_')[0]
            if eid in ecg_ids:
                valid_patients.append(path)
                # If we have enough patients, stop looking
                if num_patients is not None and len(valid_patients) >= num_patients:
                    break
        
        logging.info(f"Found {len(valid_patients)} patients with both MRI and ECG data")
        
        # If we didn't find enough patients, warn about it
        if num_patients is not None and len(valid_patients) < num_patients:
            logging.info(f"Warning: Requested {num_patients} patients but only found {len(valid_patients)}")
        
        # Load data for selected patients
        for patient_path in tqdm(valid_patients, desc="Loading patient data"):
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
            if not os.path.exists(nifti_file_path):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")
                if not os.path.exists(nifti_file_path):
                    logging.info(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load MRI data
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()
            
            H, W, Z, T = image_data.shape
            
            # Check if requested indices are valid
            if not all(z < Z for z in self.z_indices) or not all(t < T for t in self.t_indices):
                logging.info(f"Skipping {patient_path}: Requested indices out of bounds. Shape: {image_data.shape}")
                continue
            
            # Extract and normalize MRI data
            selected_data = np.stack([
                [image_data[:, :, z, t] for z in self.z_indices]
                for t in self.t_indices
            ])
            
            # Normalize each slice
            for t_idx in range(len(self.t_indices)):
                for z_idx in range(len(self.z_indices)):
                    slice_min = np.min(selected_data[t_idx, z_idx])
                    slice_max = np.max(selected_data[t_idx, z_idx])
                    selected_data[t_idx, z_idx] = (selected_data[t_idx, z_idx] - slice_min) / (slice_max - slice_min)
            
            selected_data = selected_data[..., np.newaxis]
            
            # Get corresponding ECG data but only keep specified leads
            patient_eid = patient_path.split('_')[0]
            patient_index = ecg_data['IDs'].index(patient_eid)
            ecg_median = self.median_heartbeats[patient_index][:num_leads, ...][..., np.newaxis]  # Only keep first num_leads
            
            # Store data
            self.mri_data.append(selected_data)
            self.ecg_data.append(ecg_median)
            self.patient_ids.append(patient_path)

        logging.info(f"Successfully loaded {len(self.mri_data)} patients")
        if len(self.mri_data) > 0:
            logging.info(f"MRI shape: [T={len(self.t_indices)}, Z={len(self.z_indices)}, H={self.mri_data[0].shape[2]}, W={self.mri_data[0].shape[3]}, C=1]")
            logging.info(f"ECG shape: [12, {self.ecg_data[0].shape[1]}, 1]")

    def __getitem__(self, index: int):
        return self.mri_data[index], self.ecg_data[index], self.patient_ids[index]

    def __len__(self):
        return len(self.mri_data)