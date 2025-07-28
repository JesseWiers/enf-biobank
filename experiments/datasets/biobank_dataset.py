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

def create_mosaic_augmentation(image: np.ndarray, segmentation: np.ndarray, grid_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a mosaic augmentation by dividing the image into patches and shuffling them.
    
    Args:
        image: Input image of shape (H, W, 1)
        segmentation: Input segmentation of shape (H, W)
        grid_size: Tuple of (rows, cols) specifying the grid dimensions
    """
    H, W = image.shape[:2]
    rows, cols = grid_size
    
    # Calculate patch sizes (using floor division to ensure consistent sizes)
    patch_height = H // rows
    patch_width = W // cols
    
    # Create lists to store patches with their original positions
    patches = []
    
    # Extract patches
    for i in range(rows):
        for j in range(cols):
            # Calculate boundaries ensuring we don't exceed image dimensions
            h_start = i * patch_height
            h_end = h_start + patch_height if i < rows - 1 else H
            w_start = j * patch_width
            w_end = w_start + patch_width if j < cols - 1 else W
            
            # Store patch and its position together
            patch_info = {
                'img': image[h_start:h_end, w_start:w_end].copy(),
                'seg': segmentation[h_start:h_end, w_start:w_end].copy(),
                'pos': (h_start, h_end, w_start, w_end)
            }
            patches.append(patch_info)
    
    # Shuffle the patches (not their positions)
    np.random.shuffle(patches)
    
    # Create output arrays
    augmented_image = np.zeros_like(image)
    augmented_segmentation = np.zeros_like(segmentation)
    
    # Place each patch back in its original position
    for patch in patches:
        h_start, h_end, w_start, w_end = patch['pos']
        augmented_image[h_start:h_end, w_start:w_end] = patch['img']
        augmented_segmentation[h_start:h_end, w_start:w_end] = patch['seg']
    
    return augmented_image, augmented_segmentation


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
    

class BiobankNiftiV2(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        skip_zero_seg: bool=True,
        mosaic_augment: bool=False,
        mosaic_grid_size: Tuple[int, int]=(2, 2)  # Default grid size
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.data_seg = []
        self.skip_zero_seg = skip_zero_seg
        self.mosaic_augment = mosaic_augment
        self.mosaic_grid_size = mosaic_grid_size
        patient_paths = os.listdir(root)
        
        patient_paths = sorted(os.listdir(root))

        if split == 'train':
            patient_paths = patient_paths[:4500]
            
            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
            
        elif split == 'test': 
            patient_paths = patient_paths[4500:]

            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]

            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
                
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz") 
            nifti_file_path_seg = os.path.join(root, patient_path, "seg_sa.nii.gz")

            if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
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
                    
                    # Add original image
                    self.data.append(image)
                    self.data_seg.append(image_seg)

                    # Add mosaic augmented version if requested
                    if self.mosaic_augment:
                        aug_image, aug_seg = create_mosaic_augmentation(
                            image, 
                            image_seg, 
                            self.mosaic_grid_size
                        )
                        self.data.append(aug_image)
                        self.data_seg.append(aug_seg)

        print(f"Number of datapoints: {len(self.data)}")
                

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target) for the given index"""
        return self.data[index], self.data_seg[index]

    def __len__(self):
        return len(self.data)
    
    
class BiobankNiftiLVEF(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        skip_zero_seg: bool=True,
        mosaic_augment: bool=False,
        mosaic_grid_size: Tuple[int, int]=(2, 2)  # Default grid size
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.data_seg = []
        self.skip_zero_seg = skip_zero_seg
        self.mosaic_augment = mosaic_augment
        self.mosaic_grid_size = mosaic_grid_size
        patient_paths = os.listdir(root)
        
        patient_paths = sorted(os.listdir(root))

        if split == 'train':
            patient_paths = patient_paths[:350]
            
            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
            
        elif split == 'test': 
            patient_paths = patient_paths[350:]

            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]

            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
                
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
            nifti_file_path_seg = os.path.join(root, patient_path, "cropped_seg_sa.nii.gz")

            if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")
                nifti_file_path_seg = os.path.join(root, patient_path, "seg_sa.nii.gz")

                if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
                    print(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load the NIFTI file
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                 
            nifti_image_seg = nib.load(nifti_file_path_seg)
            image_data_seg = nifti_image_seg.get_fdata()  # Shape: [H, W, Z, T]
            
            # NOTE : ADD PADDING TO MAKE IT (77, 77)
            pad_config = ((3, 3), (0, 0), (0, 0), (0, 0))
            
            image_data = np.pad(image_data,
                                        pad_width=pad_config,
                                            mode='constant',
                                            constant_values=0)

            image_data_seg = np.pad(image_data_seg,
                                        pad_width=pad_config,
                                        mode='constant',
                                        constant_values=0)
            
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
                    
                    # Add original image
                    self.data.append(image)
                    self.data_seg.append(image_seg)

                    # Add mosaic augmented version if requested
                    if self.mosaic_augment:
                        aug_image, aug_seg = create_mosaic_augmentation(
                            image, 
                            image_seg, 
                            self.mosaic_grid_size
                        )
                        self.data.append(aug_image)
                        self.data_seg.append(aug_seg)

        print(f"Number of datapoints: {len(self.data)}")
        self.indices = np.arange(len(self.data))  # Add this line to store indices
                

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target, index) for the given index"""
        return self.data[index], self.data_seg[index], self.indices[index]  # Return index as well

    def __len__(self):
        return len(self.data)
    
    
class BiobankNiftiLVEF3D(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        num_patients_train: int=None,
        num_patients_test: int=None,
        z_indices: list[int] = [0, 1],  # Added z_indices parameter
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.z_indices = sorted(z_indices)  # Ensure consistent ordering
        
        print("Using z_indices: ", z_indices)
        patient_paths = sorted(os.listdir(root))

        if split == 'train':
            patient_paths = patient_paths[:350]  # Keep LVEF dataset size
            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]
            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
        elif split == 'test': 
            patient_paths = patient_paths[350:]
            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]
            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
                
        counter = 0  # Count skipped timesteps
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")

            if not os.path.exists(nifti_file_path):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")

                if not os.path.exists(nifti_file_path):
                    print(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load the NIFTI files
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
            
            H, W, Z, T = image_data.shape
            
            # Iterate over timesteps
            for t in range(T):
                if len(self.z_indices) == 1:
                    # Handle single z-slice case
                    for z in range(Z):
                        image = image_data[:, :, z, t]
                                            
                        # Min-max normalization
                        slice_min = np.min(image)
                        slice_max = np.max(image)
                        image = (image - slice_min) / (slice_max - slice_min)
                        
                        assert np.all(image >= 0) and np.all(image <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={t}, z={z}"
                        
                        image = image[np.newaxis, ..., np.newaxis]  # Shape: [1, H, W, 1]
                        
                        self.data.append(image)
                        
                else:
                    # Handle multiple z-slices case
                    if all(z < Z for z in self.z_indices):
                        # Stack selected z-slices
                        stacked_images = np.stack([image_data[:, :, z, t] for z in self.z_indices], axis=0)
                        
                        # Normalize each slice separately
                        for i in range(len(self.z_indices)):
                            slice_min = np.min(stacked_images[i])
                            slice_max = np.max(stacked_images[i])
                            stacked_images[i] = (stacked_images[i] - slice_min) / (slice_max - slice_min)
                            assert np.all(stacked_images[i] >= 0) and np.all(stacked_images[i] <= 1), f"Normalization failed: values outside [0,1] range"
                        
                        stacked_images = stacked_images[..., np.newaxis]  # Shape: [num_z, H, W, 1]
                        
                        self.data.append(stacked_images)
                    else:
                        counter += 1

        print("Number of timesteps without all slices: ", counter)
        print(f"Number of datapoints: {len(self.data)}")

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target) for the given index"""
        return self.data[index], self.data[index]

    def __len__(self):
        return len(self.data)
    
    
class BiobankNiftiLVEF4D(Dataset):
        
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
            patient_paths = patient_paths[:4500]

            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
        elif split == 'test':
            patient_paths = patient_paths[4500:1200]

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
    
    
class BiobankNifti3DV2(Dataset):
        
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
            patient_paths = patient_paths[:4500]

            if num_patients_train != -1:
                patient_paths = patient_paths[:num_patients_train]

            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
        elif split == 'test':
            patient_paths = patient_paths[4500:]

            if num_patients_test != -1:
                patient_paths = patient_paths[:num_patients_test]

            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
        
        counter = 0
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")  # Assuming file structure

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