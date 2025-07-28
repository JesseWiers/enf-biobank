from typing import Union, Any, Sequence

import numpy as np
from torch.utils import data
import torchvision
from torch.utils.data import Dataset
import torch
from PIL import Image
import os 
import pydicom
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGLeadDataset(Dataset):
    def __init__(self, base_path: str = '/home/jwiers/deeprisk/new_codebase/enf-min-jax-version-1/data/biobank_ecg', 
                 split: str = 'train', 
                 transforms: torch.nn.Module=None, 
                 lead_idx: int = 0):
        """
        Dataset for single ECG lead reconstruction.
        
        Args:
            base_path: Base path to the ECG data directory
            split: Which split to use ('train', 'val', or 'test')
            lead_idx: Index of the lead to reconstruct (0-11)
        """
        self.lead_idx = lead_idx
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Load the pre-split data
        self.ecg_data = torch.load(f"{base_path}/ECG_leads_repeated_{split}.pt")
        self.ids = torch.load(f"{base_path}/ECG_ids_repeated_{split}.pt")

        self.transforms = transforms
        
        # Compute min and max values for the specific lead
        lead_data = self.ecg_data[:, self.lead_idx].detach().cpu().numpy()
        self.min_val = lead_data.min()
        self.max_val = lead_data.max()
        
        print(f"Loaded {len(self.ecg_data)} samples for {split} set")
        print(f"Using Lead {self.lead_names[lead_idx]}")
        print(f"Min value: {self.min_val:.4f}, Max value: {self.max_val:.4f}")
        
    def __getitem__(self, idx):
        # Get the lead, convert to numpy array, and reshape
        lead = self.ecg_data[idx, self.lead_idx].detach().cpu().numpy()

        # Reshape from (1, time_points) to (time_points, 1)
        lead = lead.reshape(-1, 1)

        # Apply min-max normalization
        lead_normalized = (lead - self.min_val) / (self.max_val - self.min_val)

        if self.transforms:
            lead_normalized = self.transforms(lead_normalized)
        
        # Return the same lead as both input and target
        return lead_normalized, lead_normalized
    
    def __len__(self):
        return len(self.ecg_data)


from torch.utils.data import DataLoader

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

def to_numpy(inp) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(inp)