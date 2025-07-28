import os
import time
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import logging

class BiobankNiftiEDESMiddleSlices(Dataset):
    def __init__(self, root: str, endpoints_csv_path: str, endpoint_name='LVEF', 
                 debug_limit=None):
        """
        Dataset for loading NIFTI images with ED and ES timepoints and middle z-slice.
        Mirrors the structure of LatentEndpointDatasetEDESMiddleSlices but for image data.
        
        Args:
            root (str): Path to directory containing patient folders with NIFTI files
            endpoints_csv_path (str): Path to CSV file with endpoints and ED/ES timepoints
            endpoint_name (str): Name of the endpoint column to use as target
            debug_limit (int, optional): Limit number of patients for debugging
        """
        print(f"Initializing dataset with root: {root}")
        print(f"Endpoints CSV: {endpoints_csv_path}")
        print("Using middle z-slice for each patient")
        
        start_time = time.time()
        
        # Load endpoints CSV
        try:
            self.endpoints_df = pd.read_csv(endpoints_csv_path)
            print(f"Loaded endpoints CSV with {len(self.endpoints_df)} rows")
        except Exception as e:
            print(f"Error loading endpoints CSV: {str(e)}")
            raise
        
        # Ensure patient ID column exists and convert to string
        patient_id_col = next((col for col in self.endpoints_df.columns 
                               if col.lower() in ['f.eid']), None)
        if patient_id_col is None:
            raise ValueError("Could not find patient ID column in endpoints CSV")
            
        self.patient_id_col = patient_id_col
        self.endpoints_df[patient_id_col] = self.endpoints_df[patient_id_col].astype(str)
        
        # Filter out patients without the target endpoint or ED/ES timepoints
        self.endpoints_df = self.endpoints_df.dropna(subset=[endpoint_name, 'ED', 'ES'])
        print(f"After filtering for valid {endpoint_name} and ED/ES: {len(self.endpoints_df)} patients")
        
        # Get all patient paths
        patient_paths = sorted(os.listdir(root))
        if debug_limit is not None:
            patient_paths = patient_paths[:debug_limit]
            
        self.data = []  # Will store (image_ed, image_es) pairs
        self.endpoints = []  # Will store endpoint values
        self.patient_ids = []  # Will store patient IDs
        
        # Process each patient
        for patient_path in tqdm(patient_paths, desc="Loading patients"):
            try:
                # Check if patient has endpoint data
                if patient_path not in set(self.endpoints_df[patient_id_col].values):
                    continue
                
                # Get ED/ES timepoints and endpoint value
                patient_data = self.endpoints_df[self.endpoints_df[patient_id_col] == patient_path]
                ed_timepoint = 0  # Fixed to 0 as in latent dataset
                es_timepoint = int(patient_data['ES'].values[0])
                endpoint_value = float(patient_data[endpoint_name].values[0])
                
                # Load NIFTI file
                nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
                if not os.path.exists(nifti_file_path):
                    print(f"Skipping {patient_path}: NIFTI file not found")
                    continue
                
                nifti_image = nib.load(nifti_file_path)
                image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                
                # Get middle z-slice
                Z = image_data.shape[2]
                middle_z = Z // 2
                
                # Extract ED and ES timepoint images from middle slice
                image_ed = image_data[:, :, middle_z, ed_timepoint]
                image_es = image_data[:, :, middle_z, es_timepoint]
                
                # Normalize each slice separately
                for image in [image_ed, image_es]:
                    slice_min = np.min(image)
                    slice_max = np.max(image)
                    image = (image - slice_min) / (slice_max - slice_min)
                    assert np.all(image >= 0) and np.all(image <= 1), "Normalization failed"
                
                # Add channel dimension
                image_ed = image_ed[..., np.newaxis]
                image_es = image_es[..., np.newaxis]
                
                # Store the data
                self.data.append((image_ed, image_es))
                self.endpoints.append(endpoint_value)
                self.patient_ids.append(patient_path)
                
            except Exception as e:
                print(f"Error processing patient {patient_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(self.data)} patients")
        print(f"Image shape: {self.data[0][0].shape}")  # Show shape of first ED image
        
        end_time = time.time()
        print(f"Dataset initialization completed in {end_time - start_time:.2f} seconds")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (patient_id, images, endpoint_value) where:
                - patient_id is the patient identifier
                - images is a tuple (image_ed, image_es) of normalized images with shape [H, W, 1]
                - endpoint_value is the LVEF value
        """
        return self.patient_ids[idx], self.data[idx], self.endpoints[idx]

def collate_fn_images(batch):
    """
    Custom collate function for batching the images.
    """
    patient_ids, image_pairs, endpoints = zip(*batch)
    
    # Split ED and ES images
    images_ed, images_es = zip(*image_pairs)
    
    # Stack into batches
    images_ed = np.stack(images_ed)  # [B, H, W, 1]
    images_es = np.stack(images_es)  # [B, H, W, 1]
    endpoints = np.array(endpoints)
    
    # Convert to torch tensors if needed
    images_ed = torch.from_numpy(images_ed).float()
    images_es = torch.from_numpy(images_es).float()
    endpoints = torch.from_numpy(endpoints).float()
    
    return patient_ids, (images_ed, images_es), endpoints

