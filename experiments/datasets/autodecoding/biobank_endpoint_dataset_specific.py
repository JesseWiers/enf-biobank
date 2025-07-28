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
import pandas as pd # Added for pandas
from PIL import Image
from torch.utils.data import DataLoader
import torchvision

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
    endpoint_name: str = None
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

    if dataset_name == "endpoints_4d_specific":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = EndpointDatasetSpecific(
            root='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', 
            endpoint_name=endpoint_name,
            endpoints_dir='/projects/prjs1252/data_jesse_final_v3/endpoints/',
            transform=transforms,
            target_transform=transforms,
            num_patients=num_patients, 
            z_indices=z_indices, 
            t_indices=t_indices,
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


class EndpointDatasetSpecific(Dataset):
        
    def __init__(
        self,
        root: str,
        endpoint_name: str,
        endpoints_dir: str = "/projects/prjs1252/data_jesse_final_v3/endpoints/",
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients: int=None,
        z_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        t_indices: list[int] = [0, 1],
        random_seed: int = 42
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.z_indices = sorted(z_indices)
        self.t_indices = sorted(t_indices)
        self.patient_ids = []
        
        logging.info(f"Using z_indices: {z_indices}")
        logging.info(f"Using t_indices: {t_indices}")
        
        # Load endpoint EIDs
        endpoint_path = os.path.join(endpoints_dir, f"{endpoint_name}_eids.csv")
        cases_df = pd.read_csv(endpoint_path)
        case_eid_col = next(col for col in cases_df.columns if col.lower() in ['f.eid', 'eid'])
        endpoint_eids = set(cases_df[case_eid_col].astype(str))
        
        # Get and filter patient paths
        patient_paths = sorted(os.listdir(root))
        patient_paths = [p for p in patient_paths if p in endpoint_eids]
        
        total_initial_patients = len(patient_paths)
        logging.info(f"Total number of patients found for endpoint {endpoint_name}: {total_initial_patients}")

        # Limit number of patients if specified
        if num_patients is not None and num_patients != -1:
            patient_paths = patient_paths[:num_patients]
            logging.info(f"Limiting to {num_patients} patients")

        counter = 0
        patients_with_missing_files = 0
        patients_with_invalid_indices = 0
        excluded_shape_patients = 0  # Add this
        shape_mismatches = {}  # Add this
        
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
            
            # Check image dimensions (H, W)
            if H != 71 or W != 77:
                excluded_shape_patients += 1  # Add this
                shape_key = f"{H}x{W}"
                shape_mismatches[shape_key] = shape_mismatches.get(shape_key, 0) + 1
                logging.info(f"Excluded patient {patient_path} due to incorrect shape: {(H, W)} (expected: 71x77)")
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
        logging.info(f"- Patients with incorrect shape: {excluded_shape_patients}")  # Add this
        logging.info("Shape mismatches encountered:")  # Add this
        for shape, count in shape_mismatches.items():  # Add this
            logging.info(f"  Shape {shape}: {count} patients")
        logging.info(f"- Final patients loaded: {len(self.data)}")
        
        if len(self.data) > 0:
            logging.info(f"Data shape: [T={len(self.t_indices)}, Z={len(self.z_indices)}, H={self.data[0].shape[2]}, W={self.data[0].shape[3]}, C=1]")

    def __getitem__(self, index: int):
        return self.data[index], self.patient_ids[index]

    def __len__(self):
        return len(self.data)