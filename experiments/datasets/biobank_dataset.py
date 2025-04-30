from typing import Any

import numpy as np
from torch.utils.data import Dataset
import torch
import os 
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np


class BiobankNifti(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        skip_zero_seg: bool=True
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.data_seg = []
        self.skip_zero_seg = skip_zero_seg
        patient_paths = os.listdir(root)
        
        patient_paths = sorted(os.listdir(root))

        if split == 'train':
            patient_paths = patient_paths[:1150]
            
            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
            
        elif split == 'test': 
            patient_paths = patient_paths[1150:1200]

            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]

            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
                
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "sa_cropped.nii.gz") 
            nifti_file_path_seg = os.path.join(root, patient_path, "seg_sa_cropped.nii.gz")

            if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
                print(f"Skipping {patient_path}: NIFTI files not found.")
                continue
            
            # Load the NIFTI file
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                 
            nifti_image_seg = nib.load(nifti_file_path_seg)
            image_data_seg = nifti_image_seg.get_fdata()  # Shape: [H, W, Z, T]
            
            H, W, Z, T = image_data.shape
            
            # Iterate over timesteps
            for t in range(T):
                for z in range(Z):
                    image = image_data[:, :, z, t] # Load image 
                    image_seg = image_data_seg[:, :, z, t] # Load segmentation
                    
                    # if image_seg is all zeros, skip
                    if np.all(image_seg == 0) and self.skip_zero_seg:
                        continue
                    
                    # Min-max normalization per slice
                    slice_min = np.min(image)
                    slice_max = np.max(image)
                    
                    image = (image - slice_min) / (slice_max - slice_min)
                
                    # Assert all values are between 0 and 1
                    assert np.all(image >= 0) and np.all(image <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={t}, z={z}"
             
                    image = image[..., np.newaxis] # Add axis
                                        
                    self.data.append(image)
                    self.data_seg.append(image_seg)

        print(f"Number of datapoints: {len(self.data)}")
                

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target) for the given index"""
        return self.data[index], self.data_seg[index]

    def __len__(self):
        return len(self.data)
    
    
class BiobankNifti3D(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',  
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        z_indices: list[int] = [0, 1],
    ):

        self.transform = transform
        self.target_transform = target_transform
        self.z_indices = sorted(z_indices)  # Ensure consistent ordering
        self.data = []

        print("Using z_indices: ", z_indices)
        
        patient_paths = os.listdir(root)

        patient_paths = sorted(os.listdir(root))

        if split == 'train':
            patient_paths = patient_paths[:1150]

            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
        elif split == 'test':
            patient_paths = patient_paths[1150:1200]

            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]

            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
        
        counter = 0
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "sa_cropped.nii.gz")  # Assuming file structure

            if not os.path.exists(nifti_file_path):
                print(f"Skipping {patient_path}: NIFTI file not found.")
                continue
            
            # Load the NIFTI file
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
            
            H, W, Z, T = image_data.shape
            # print(f"Loaded {patient_path} with shape: {image_data.shape}")

            # Iterate over timesteps
            for t in range(T):
                # Check if all requested z_indices exist
                
                if len(self.z_indices) == 1:
                    for z in range(Z):
                        image = image_data[:, :, z, t]
                        
                        # Min-max normalization per slice
                        slice_min = np.min(image)
                        slice_max = np.max(image)
                        
                        image = (image - slice_min) / (slice_max - slice_min)
                        
                        # Assert all values are between 0 and 1
                        assert np.all(image >= 0) and np.all(image <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={t}, z={z}"
                        
                        image = image[..., np.newaxis]
                        image = image[np.newaxis, ...]
                        self.data.append(image)
                    
                else: 
                    if all(z < Z for z in self.z_indices):
                        # Stack along first axis (axis=0) instead of last axis
                        stacked_images = np.stack([image_data[:, :, z, t] for z in self.z_indices], axis=0)
                        
                        # Normalize each slice separately
                        for i in range(len(self.z_indices)):
                            # Min-max normalization
                            slice_min = np.min(stacked_images[i])
                            slice_max = np.max(stacked_images[i])
                            
                            
                            stacked_images[i] = (stacked_images[i] - slice_min) / (slice_max - slice_min)
                         
                            # Assert all values are between 0 and 1
                            assert np.all(stacked_images[i] >= 0) and np.all(stacked_images[i] <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={t}, z={self.z_indices[i]}"

                        # Add a channel axis: [Z_indices, H, W, 1]
                        stacked_images = stacked_images[..., np.newaxis]
                        
                        self.data.append(stacked_images)
                    else:
                        counter += 1
                        # print(f"Skipping timestep {t} for {patient_path}: missing z_indices.")

        print("Number of timesteps without all slices: ", counter)
        print(f"Number of datapoints: {len(self.data)}")
                

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target) for the given index"""
        return self.data[index], self.data[index]

    def __len__(self):
        return len(self.data)