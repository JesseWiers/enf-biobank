from typing import Any, Tuple

import numpy as np
from torch.utils.data import Dataset
import torch
import os 
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import logging


class EndpointDataset(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        z_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],  # Specify which z slices to use
        t_indices: list[int] = [0, 1],  # Specify which time points to use,
        random_seed: int = 42  # Add random seed parameter for reproducibility
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.z_indices = sorted(z_indices)  # Ensure consistent ordering
        self.t_indices = sorted(t_indices)  # Ensure consistent ordering
        self.patient_ids = []  # Add list to store patient IDs
        
        logging.info(f"Using z_indices: {z_indices}")
        logging.info(f"Using t_indices: {t_indices}")
        
        # Get and sort patient paths first
        patient_paths = sorted(os.listdir(root))
        total_initial_patients = len(patient_paths)
        logging.info(f"Total number of patients found in directory: {total_initial_patients}")

        # Random shuffle with fixed seed for reproducibility
        np.random.seed(random_seed)
        np.random.shuffle(patient_paths)
        logging.info(f"Randomly shuffled patient paths with seed {random_seed}")

        if split == 'train':
            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]
                logging.info(f"Limiting to {num_patients_train} training patients")
            logging.info(f"Number of patients selected for training: {len(patient_paths)}")
        # elif split == 'test': 
        #     patient_paths = patient_paths[350:]
        #     if num_patients_test != -1:
        #         patient_paths = patient_paths[:num_patients_test]
        #     logging.info(f"Number of patients selected for testing: {len(patient_paths)}")
                
        counter = 0  # Count skipped cases
        patients_with_missing_files = 0
        patients_with_invalid_indices = 0
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
            if not os.path.exists(nifti_file_path):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")
                if not os.path.exists(nifti_file_path):
                    patients_with_missing_files += 1
                    logging.debug(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load the NIFTI files
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
            
            H, W, Z, T = image_data.shape
            
            # Check if requested indices are valid
            if not all(z < Z for z in self.z_indices) or not all(t < T for t in self.t_indices):
                patients_with_invalid_indices += 1
                logging.debug(f"Skipping {patient_path}: Requested indices out of bounds. Shape: {image_data.shape}")
                counter += 1
                continue
            
            # Extract selected t-points and z-slices (now T first, then Z)
            selected_data = np.stack([
                [image_data[:, :, z, t] for z in self.z_indices]
                for t in self.t_indices
            ])  # Shape: [num_t, num_z, H, W]
            
            # Normalize each slice separately
            for t_idx in range(len(self.t_indices)):
                for z_idx in range(len(self.z_indices)):
                    slice_min = np.min(selected_data[t_idx, z_idx])
                    slice_max = np.max(selected_data[t_idx, z_idx])
                    selected_data[t_idx, z_idx] = (selected_data[t_idx, z_idx] - slice_min) / (slice_max - slice_min)
                    assert np.all(selected_data[t_idx, z_idx] >= 0) and np.all(selected_data[t_idx, z_idx] <= 1), \
                        f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={self.t_indices[t_idx]}, z={self.z_indices[z_idx]}"
            
            # Add channel dimension: [num_t, num_z, H, W, 1]
            selected_data = selected_data[..., np.newaxis]
            
            self.data.append(selected_data)
            self.patient_ids.append(patient_path)

        logging.info(f"Dataset loading summary:")
        logging.info(f"- Initial patients: {total_initial_patients}")
        logging.info(f"- Patients with missing files: {patients_with_missing_files}")
        logging.info(f"- Patients with invalid indices: {patients_with_invalid_indices}")
        logging.info(f"- Final patients loaded: {len(self.data)}")
        
        if len(self.data) > 0:
            logging.info(f"Data shape: [T={len(self.t_indices)}, Z={len(self.z_indices)}, H={self.data[0].shape[2]}, W={self.data[0].shape[3]}, C=1]")

    def __getitem__(self, index: int):
        return self.data[index], self.patient_ids[index]

    def __len__(self):
        return len(self.data)
    
    
   