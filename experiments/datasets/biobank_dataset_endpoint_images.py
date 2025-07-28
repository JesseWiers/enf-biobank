import os
import time
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import logging

def get_middle_indices(total_length, n_indices):
    """
    Get the middle n_indices from a sequence of length total_length.
    
    Args:
        total_length (int): Total number of indices (e.g., number of z slices)
        n_indices (int): Number of middle indices to select
        
    Returns:
        list: Selected middle indices
    """
    if n_indices > total_length:
        raise ValueError(f"Cannot select {n_indices} indices from sequence of length {total_length}")
        
    # Calculate the middle index
    middle = total_length // 2
    
    # Calculate how many indices we need on each side of the middle
    indices_each_side = (n_indices - 1) // 2
    
    # Handle odd/even cases
    if n_indices % 2 == 0:
        # Even number of indices: select middle-1 and middle as center points
        start = middle - indices_each_side - 1
        end = middle + indices_each_side + 1
    else:
        # Odd number of indices: select middle as center point
        start = middle - indices_each_side
        end = middle + indices_each_side + 1
        
    return list(range(start, end))

class ImageEndpointDataset(Dataset):
    def __init__(self, nifti_root, endpoint_name, endpoints_dir="/projects/prjs1252/data_jesse_final_v3/endpoints/",
                 z_indices=-1, t_indices=-1, debug_limit=None, random_seed=42):
        np.random.seed(random_seed)
        self.nifti_root = nifti_root
        
        # Load case and control EIDs
        endpoint_path = os.path.join(endpoints_dir, f"{endpoint_name}_eids.csv")
        healthy_path = os.path.join(endpoints_dir, "healthy_matched_final_eids.csv")
        
        # First process cases
        cases_df = pd.read_csv(endpoint_path)
        case_eid_col = next(col for col in cases_df.columns if col.lower() in ['f.eid', 'eid'])
        cases = set(cases_df[case_eid_col].astype(str))
        
        # Check only case patients first
        available_cases = []
        excluded_cases = 0
        
        for patient_id in tqdm(cases, desc="Filtering cases"):
            nifti_path = os.path.join(nifti_root, patient_id, "cropped_sa.nii.gz")
            if not os.path.exists(nifti_path):
                continue
                
            try:
                nifti_image = nib.load(nifti_path)
                image_shape = nifti_image.shape  # [H, W, Z, T]
                
                # Check image dimensions (H, W)
                if image_shape[0] != 71 or image_shape[1] != 77:
                    logging.info(f"Excluded case patient {patient_id} due to incorrect shape: {image_shape[:2]}")
                    excluded_cases += 1
                    continue
                
                # Check Z dimension
                if image_shape[2] >= 5:
                    available_cases.append(patient_id)
                else:
                    logging.info(f"Excluded case patient {patient_id} with Z={image_shape[2]}")
                    excluded_cases += 1
                    
            except Exception as e:
                logging.warning(f"Error loading NIFTI for case patient {patient_id}: {str(e)}")
                continue
        
        n_cases = len(available_cases)
        logging.info(f"Found {n_cases} valid cases, excluded {excluded_cases} cases")
        
        if n_cases == 0:
            raise ValueError("No valid cases found in NIFTI directory")
        
        # Now load and process only the needed number of controls
        healthy_df = pd.read_csv(healthy_path)
        healthy_eid_col = next(col for col in healthy_df.columns if col.lower() in ['f.eid', 'eid'])
        all_healthy_eids = set(healthy_df[healthy_eid_col].astype(str))
        
        # Randomly sample 2x the number of cases we need to account for potential exclusions
        n_controls_to_check = min(2 * n_cases, len(all_healthy_eids))
        controls_to_check = set(np.random.choice(list(all_healthy_eids), size=n_controls_to_check, replace=False))
        
        # Check Z dimension for sampled controls
        available_controls = []
        excluded_controls = 0
        
        for patient_id in tqdm(controls_to_check, desc="Filtering controls"):
            if len(available_controls) >= n_cases:
                break  # Stop once we have enough controls
                
            nifti_path = os.path.join(nifti_root, patient_id, "cropped_sa.nii.gz")
            if not os.path.exists(nifti_path):
                continue
                
            try:
                nifti_image = nib.load(nifti_path)
                image_shape = nifti_image.shape
                
                if image_shape[0] != 71 or image_shape[1] != 77:
                    excluded_controls += 1
                    continue
                
                if image_shape[2] >= 5:
                    available_controls.append(patient_id)
                else:
                    excluded_controls += 1
                    
            except Exception as e:
                continue
        
        # If we don't have enough controls, sample more
        if len(available_controls) < n_cases:
            remaining_controls = all_healthy_eids - controls_to_check
            while len(available_controls) < n_cases and remaining_controls:
                additional_to_check = set(np.random.choice(
                    list(remaining_controls),
                    size=min(n_cases - len(available_controls), len(remaining_controls)),
                    replace=False
                ))
                
                for patient_id in tqdm(additional_to_check, desc="Checking additional controls"):
                    if len(available_controls) >= n_cases:
                        break
                        
                    nifti_path = os.path.join(nifti_root, patient_id, "cropped_sa.nii.gz")
                    if not os.path.exists(nifti_path):
                        continue
                        
                    try:
                        nifti_image = nib.load(nifti_path)
                        image_shape = nifti_image.shape
                        
                        if image_shape[0] != 71 or image_shape[1] != 77:
                            excluded_controls += 1
                            continue
                        
                        if image_shape[2] >= 5:
                            available_controls.append(patient_id)
                        else:
                            excluded_controls += 1
                            
                    except Exception as e:
                        continue
                
                remaining_controls -= additional_to_check
        
        # Take only the number of controls we need
        available_controls = available_controls[:n_cases]
        
        logging.info(f"Found {len(available_controls)} valid controls, excluded {excluded_controls} controls")
        
        # Combine cases and controls
        self.patient_ids = available_cases + available_controls
        self.labels = [1] * len(available_cases) + [0] * len(available_controls)
        
        # Process indices
        first_nifti = nib.load(os.path.join(nifti_root, self.patient_ids[0], "cropped_sa.nii.gz"))
        _, _, z_dim, t_dim = first_nifti.shape
        
        self.z_indices = z_indices
        if t_indices == -1:
            self.t_indices = list(range(t_dim))
        else:
            self.t_indices = t_indices if isinstance(t_indices, list) else [t_indices]
            
        logging.info(f"Dataset initialized with {len(self.patient_ids)} patients ({len(available_cases)} cases + {len(available_controls)} controls)")

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        
        # Load NIFTI file
        nifti_path = os.path.join(self.nifti_root, patient_id, "cropped_sa.nii.gz")
        nifti_image = nib.load(nifti_path)
        image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
        
        # Get z-indices based on this patient's Z dimension
        Z = image_data.shape[2]
        if isinstance(self.z_indices, int):
            if self.z_indices == -1:
                z_indices_to_use = list(range(Z))
            else:
                z_indices_to_use = get_middle_indices(Z, self.z_indices)
        else:
            z_indices_to_use = self.z_indices
            
        # Collect selected slices
        selected_images = []
        for t_idx in self.t_indices:
            for z_idx in z_indices_to_use:
                if z_idx >= Z:
                    continue
                    
                # Get slice
                image = image_data[:, :, z_idx, t_idx]
                
                # Normalize slice
                slice_min = np.min(image)
                slice_max = np.max(image)
                image = (image - slice_min) / (slice_max - slice_min)
                
                # Add channel dimension
                image = image[..., np.newaxis]
                selected_images.append(image)
        
        if not selected_images:
            raise ValueError(f"No valid slices selected for patient {patient_id}")
        
        # Stack all selected slices along channel dimension
        images = np.stack(selected_images, axis=-1)  # [H, W, 1, N] where N = num_timepoints * num_z_slices
        
        return patient_id, images, label

def collate_fn_images_endpoint(batch):
    patient_ids, images, labels = zip(*batch)
    
    # Debug: Log shapes of all images in batch
    logging.debug("Batch image shapes:")
    for pid, img in zip(patient_ids, images):
        logging.debug(f"Patient {pid}: shape {img.shape}")
    
    try:
        stacked_images = np.stack(images)
    except ValueError as e:
        logging.error("\nShape mismatch in batch!")
        shapes = [img.shape for img in images]
        unique_shapes = set(shapes)
        logging.error(f"Found {len(unique_shapes)} different shapes in batch:")
        for shape in unique_shapes:
            matching_pids = [pid for pid, img in zip(patient_ids, images) if img.shape == shape]
            logging.error(f"Shape {shape}: Patients {matching_pids}")
        raise

    return patient_ids, stacked_images, np.array(labels)

