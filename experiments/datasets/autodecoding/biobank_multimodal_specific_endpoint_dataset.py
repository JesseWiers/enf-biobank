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
    endpoint_name: str = None,  # Add this parameter
    num_patients: int = None,
    seed: int = 42,
    z_indices: tuple = (0, 1, 2, 3, 4, 5, 6, 7),
    t_indices: tuple = (0, 1),
    num_leads: int = 12
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

    if dataset_name == "multi_modal_specific":
        train_dset = EndpointDatasetMultiModalSpecific(
            root='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', 
            ecg_path='/projects/prjs1252/data_jesse_final_v3/ECGs_median_leads.pth',
            endpoint_name=endpoint_name,  # Add this parameter
            num_patients=num_patients, 
            z_indices=z_indices, 
            t_indices=t_indices,
            num_leads=num_leads
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

def image_to_numpy(image: Image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image) / 255

class EndpointDatasetMultiModalSpecific(Dataset):
    """Dataset for loading both MRI and ECG data for specific endpoints.
    
    This dataset:
    1. Filters patients by specific endpoint
    2. Ensures patients have both MRI and ECG data
    3. Handles data normalization and shape requirements
    4. Returns matched MRI and ECG data for each patient
    """
    def __init__(
        self,
        root: str,  # MRI data root
        ecg_path: str,  # Path to ECG data
        endpoint_name: str,  # Name of the endpoint (e.g., 'cardiomyopathy')
        endpoints_dir: str = "/projects/prjs1252/data_jesse_final_v3/endpoints/",
        num_patients: int = None,
        z_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        t_indices: list[int] = [0, 1],
        num_leads: int = 12,
        random_seed: int = 42,
    ):
        self.mri_data = []
        self.ecg_data = []
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
        
        # Load ECG data
        ecg_data = torch.load(ecg_path)
        ecg_ids = set(ecg_data['IDs'])
        
        # Find patients that have both ECG data and the specific endpoint
        valid_patient_eids = endpoint_eids & ecg_ids
        logging.info(f"Found {len(valid_patient_eids)} patients with endpoint {endpoint_name} and ECG data")
        
        # Get and filter patient paths
        patient_paths = sorted(os.listdir(root))
        patient_paths = [p for p in patient_paths if p.split('_')[0] in valid_patient_eids]
        
        total_initial_patients = len(patient_paths)
        logging.info(f"Total number of patients found for endpoint {endpoint_name}: {total_initial_patients}")

        # Limit number of patients if specified
        if num_patients is not None and num_patients != -1:
            patient_paths = patient_paths[:num_patients]
            logging.info(f"Limiting to {num_patients} patients")

        # Tracking statistics
        patients_with_missing_files = 0
        patients_with_invalid_indices = 0
        excluded_shape_patients = 0
        shape_mismatches = {}
        
        for patient_path in tqdm(patient_paths, desc=f"Loading {endpoint_name} patient data"):
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
            if not os.path.exists(nifti_file_path):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")
                if not os.path.exists(nifti_file_path):
                    patients_with_missing_files += 1
                    logging.debug(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load MRI data
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()
            
            H, W, Z, T = image_data.shape
            
            # Check if requested indices are valid
            if not all(z < Z for z in self.z_indices) or not all(t < T for t in self.t_indices):
                patients_with_invalid_indices += 1
                logging.debug(f"Skipping {patient_path}: Requested indices out of bounds. Shape: {image_data.shape}")
                continue
            
            # Check image dimensions (H, W)
            if H != 71 or W != 77:
                excluded_shape_patients += 1
                shape_key = f"{H}x{W}"
                shape_mismatches[shape_key] = shape_mismatches.get(shape_key, 0) + 1
                logging.info(f"Excluded patient {patient_path} due to incorrect shape: {(H, W)} (expected: 71x77)")
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
            
            # Get corresponding ECG data
            patient_eid = patient_path.split('_')[0]
            patient_index = ecg_data['IDs'].index(patient_eid)
            ecg_median = ecg_data['median_heartbeats'][patient_index][:num_leads, ...][..., np.newaxis]
            
            # Store data
            self.mri_data.append(selected_data)
            self.ecg_data.append(ecg_median)
            self.patient_ids.append(patient_path)

        # Log summary statistics
        logging.info(f"Dataset loading summary for endpoint {endpoint_name}:")
        logging.info(f"- Initial patients: {total_initial_patients}")
        logging.info(f"- Patients with missing files: {patients_with_missing_files}")
        logging.info(f"- Patients with invalid indices: {patients_with_invalid_indices}")
        logging.info(f"- Patients with incorrect shape: {excluded_shape_patients}")
        logging.info("Shape mismatches encountered:")
        for shape, count in shape_mismatches.items():
            logging.info(f"  Shape {shape}: {count} patients")
        logging.info(f"- Final patients loaded: {len(self.mri_data)}")
        
        if len(self.mri_data) > 0:
            logging.info(f"MRI shape: [T={len(self.t_indices)}, Z={len(self.z_indices)}, H={self.mri_data[0].shape[2]}, W={self.mri_data[0].shape[3]}, C=1]")
            logging.info(f"ECG shape: [{num_leads}, {self.ecg_data[0].shape[1]}, 1]")

    def __getitem__(self, index: int):
        return self.mri_data[index], self.ecg_data[index], self.patient_ids[index]

    def __len__(self):
        return len(self.mri_data)