import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
from skimage.transform import resize  # Add this import

class CardiacPatchDataset(Dataset):
    def __init__(self, nifti_dir, patch_size=16, target_size=80):
        """
        Dataset for extracting patches from cardiac MRI slices
        
        Args:
            nifti_dir (str): Directory containing NIFTI files
            patch_size (int): Size of patches (default 16x16)
            target_size (int): Size to resize images to (default 80x80)
        """
        self.nifti_dir = Path(nifti_dir)
        self.patch_size = patch_size
        self.target_size = target_size
        self.samples = []
        
        # Calculate grid size
        self.grid_size = target_size // patch_size
        
        # Build dataset
        self._build_dataset()
    
    def _build_dataset(self):
        """Build list of all patches to extract"""
        for patient_dir in self.nifti_dir.glob("*"):
            if not patient_dir.is_dir():
                continue
                
            nifti_path = patient_dir / "cropped_sa.nii.gz"
            if not nifti_path.exists():
                continue
            
            # Load NIFTI
            nifti_img = nib.load(str(nifti_path))
            img_data = nifti_img.get_fdata()  # Shape: [H, W, Z, T]
            
            # For each slice and timepoint
            for z in range(img_data.shape[2]):
                for t in range(img_data.shape[3]):
                    self.samples.append({
                        'patient_id': patient_dir.name,
                        'z': z,
                        't': t,
                    })
    
    def _resize_slice(self, slice_data):
        """Resize slice to target size using bilinear interpolation"""
        return resize(slice_data, 
                 (self.target_size, self.target_size),
                 order=1,  # 1 = bilinear interpolation
                 anti_aliasing=True,
                 mode='reflect'  # boundary handling mode
        )
    
    def _extract_patches(self, padded_slice):
        """Extract patches from padded slice"""
        patches = []
        positions = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract patch
                start_h = i * self.patch_size
                start_w = j * self.patch_size
                patch = padded_slice[start_h:start_h + self.patch_size,
                                   start_w:start_w + self.patch_size]
                
                # Store patch and its position
                patches.append(patch)
                positions.append([i, j])  # Grid position
        
        return np.array(patches), np.array(positions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load slice
        nifti_path = self.nifti_dir / sample['patient_id'] / "cropped_sa.nii.gz"
        nifti_img = nib.load(str(nifti_path))
        img_data = nifti_img.get_fdata()
        
        # Get specific slice
        slice_data = img_data[:, :, sample['z'], sample['t']]
        
        # Min-max normalization per slice (before resizing)
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        
        # Assert all values are between 0 and 1
        assert np.all(slice_data >= 0) and np.all(slice_data <= 1), f"Normalization failed: values outside [0,1] range"
        
        # Resize slice to target size
        resized_slice = self._resize_slice(slice_data)
        
        # Extract patches and positions
        patches, positions = self._extract_patches(resized_slice)
        
        # No need for additional normalization since patches are already normalized

        return {
            'patches': torch.FloatTensor(patches),  # Shape: [25, patch_size, patch_size]
            'positions': torch.FloatTensor(positions),  # Shape: [25, 2]
            'metadata': {
                'patient_id': sample['patient_id'],
                'z': sample['z'],
                't': sample['t']
            }
        }