import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # Add at the top with other imports
import h5py
import pandas as pd
import numpy as np
import os
import jax.numpy as jnp
import time
import traceback
import logging

class LatentEndpointDatasetAutodecodingMultimodal(Dataset):
    def __init__(self, healthy_hdf5_path, endpoint_name, endpoints_dir="/projects/prjs1252/data_jesse_final_v3/endpoints/",
                 z_indices=-1, t_indices=-1, debug_limit=None, random_seed=42, latents_per_dim=8):
        """
        Dataset for autodecoding latent representations with balanced endpoint cases and controls.
        
        Args:
            healthy_hdf5_path (str): Path to HDF5 file containing healthy patient latents
            endpoint_name (str): Name of the endpoint (e.g., 'cardiomyopathy')
            endpoints_dir (str): Directory containing endpoint CSV files
            z_indices (list or int): Z indices to include (max latents_per_dim) or -1 for all
            t_indices (list or int): Time indices to include (max latents_per_dim) or -1 for all
            latents_per_dim (int): Number of latents per dimension (e.g., 8 for 4096 total latents)
        """
        np.random.seed(random_seed)
        self.healthy_hdf5_path = healthy_hdf5_path
        
        # Construct endpoint HDF5 path
        base_dir = "/projects/prjs1252/data_jesse_final_v3/autodecoding_latent_datasets_multimodal"
        total_latents = latents_per_dim ** 4
        self.endpoint_hdf5_path = os.path.join(
            base_dir, 
            f"{total_latents}latents_32dim_{endpoint_name}_endpoint_200epochs_pretrain_100patients_1920_epochs.h5"
        )
        
        # Store dimensions
        self.latents_per_dim = latents_per_dim
        self.total_latents = total_latents
        
        # Load case and control EIDs
        endpoint_path = os.path.join(endpoints_dir, f"{endpoint_name}_eids.csv")
        cases_df = pd.read_csv(endpoint_path)
        case_eid_col = next(col for col in cases_df.columns if col.lower() in ['f.eid', 'eid'])
        cases = set(cases_df[case_eid_col].astype(str))
        
        # First process cases from endpoint HDF5
        with h5py.File(self.endpoint_hdf5_path, 'r') as f:
            available_cases = []
            for patient_id in tqdm(cases, desc="Processing cases"):
                if patient_id in f['patients']:
                    available_cases.append(patient_id)
                    
        n_cases = len(available_cases)
        if n_cases == 0:
            raise ValueError("No valid cases found in endpoint HDF5 file")
            
        logging.info(f"Found {n_cases} cases in endpoint HDF5")
        
        # Now process controls from healthy HDF5
        with h5py.File(healthy_hdf5_path, 'r') as f:
            all_controls = list(f['patients'].keys())
            # Randomly sample exactly n_cases controls
            available_controls = list(np.random.choice(all_controls, size=n_cases, replace=False))
            
        logging.info(f"Selected {len(available_controls)} controls to match cases")
        
        # Combine cases and controls
        self.patient_ids = available_cases + available_controls
        self.labels = [1] * len(available_cases) + [0] * len(available_controls)
        
        # Process indices
        if isinstance(z_indices, int):
            if z_indices == -1:
                self.z_indices = list(range(latents_per_dim))
            else:
                self.z_indices = get_middle_indices(latents_per_dim, z_indices)
        else:
            self.z_indices = z_indices
            
        if isinstance(t_indices, int):
            if t_indices == -1:
                self.t_indices = list(range(latents_per_dim))
            else:
                self.t_indices = get_middle_indices(latents_per_dim, t_indices)
        else:
            self.t_indices = t_indices
            
        # Validate indices
        if max(self.z_indices) >= latents_per_dim or max(self.t_indices) >= latents_per_dim:
            raise ValueError(f"Indices must be less than latents_per_dim ({latents_per_dim})")
            
        logging.info(f"Dataset initialized with {len(self.patient_ids)} patients "
                    f"({len(available_cases)} cases + {len(available_controls)} controls)")
        logging.info(f"Using z_indices: {self.z_indices}")
        logging.info(f"Using t_indices: {self.t_indices}")

        # Calculate and log the number of latents being used
        total_x_latents = self.latents_per_dim
        total_y_latents = self.latents_per_dim
        total_z_latents = len(self.z_indices)
        total_t_latents = len(self.t_indices)
        
        latents_per_patient = total_x_latents * total_y_latents * total_z_latents * total_t_latents
        
        logging.info(f"\nLatent representation details:")
        logging.info(f"- Original latents per dimension: {self.latents_per_dim}")
        logging.info(f"- Original total latents: {self.total_latents}")
        logging.info(f"- Selected dimensions:")
        logging.info(f"  * X: all {total_x_latents} latents")
        logging.info(f"  * Y: all {total_y_latents} latents")
        logging.info(f"  * Z: {total_z_latents} latents {self.z_indices}")
        logging.info(f"  * T: {total_t_latents} latents {self.t_indices}")
        logging.info(f"- Final latents per patient: {latents_per_patient}\n")

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        
        # Determine which HDF5 file to use based on label
        hdf5_path = self.endpoint_hdf5_path if label == 1 else self.healthy_hdf5_path
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load p, c, g components
            p = jnp.array(f[f'patients/{patient_id}/p'][:])
            c = jnp.array(f[f'patients/{patient_id}/c'][:])
            g = jnp.array(f[f'patients/{patient_id}/g'][:])
            
            # Reshape each component to 4D grid
            grid_shape = (self.latents_per_dim,) * 4  # (8, 8, 8, 8) for 4096 latents
            p = p.reshape(grid_shape + (-1,))
            c = c.reshape(grid_shape + (-1,))
            g = g.reshape(grid_shape + (-1,))
            
            # Select z and t indices for each component
            p = p[:, :, self.z_indices][:, :, :, self.t_indices]
            c = c[:, :, self.z_indices][:, :, :, self.t_indices]
            g = g[:, :, self.z_indices][:, :, :, self.t_indices]
            
            # Flatten spatial dimensions
            p = p.reshape(-1, p.shape[-1])
            c = c.reshape(-1, c.shape[-1])
            g = g.reshape(-1, g.shape[-1])
            
            return patient_id, (p, c, g), label

def collate_fn(batch):
    try:
        patient_ids, z_tuples, labels = zip(*batch)
        
        # Separate p, c, g from z_tuples
        p_list = [z[0] for z in z_tuples]
        c_list = [z[1] for z in z_tuples]
        g_list = [z[2] for z in z_tuples]
        
        # Stack each component
        p_batched = jnp.stack(p_list)
        c_batched = jnp.stack(c_list)
        g_batched = jnp.stack(g_list)
        
        # Create batched z_tuple
        z_batched = (p_batched, c_batched, g_batched)
        
        # Convert labels to array
        labels_array = jnp.array(labels)
        
        return patient_ids, z_batched, labels_array
    except Exception as e:
        logging.error(f"Error in collate_fn: {str(e)}")
        return [], (jnp.array([]), jnp.array([]), jnp.array([])), jnp.array([])

def create_dataloaders(healthy_hdf5_path, endpoint_name, batch_size=16, 
                      train_split=0.75, val_split=0.15,  # Modified split ratios
                      num_workers=0, debug_limit=None, random_seed=42,
                      z_indices=-1, t_indices=-1):
    """
    Create train, validation, and test dataloaders with a 75/15/15 split.
    
    Args:
        hdf5_path (str): Path to HDF5 file containing latent representations
        endpoint_name (str): Name of the endpoint (e.g., 'sudden_cardiac_death')
        batch_size (int): Batch size for dataloaders
        train_split (float): Fraction of data to use for training (default: 0.75)
        val_split (float): Fraction of data to use for validation (default: 0.15)
        num_workers (int): Number of workers for DataLoader
        debug_limit (int, optional): Limit number of patients for debugging
        random_seed (int): Random seed for reproducibility
        z_indices (list or int): Indices of z-slices to include
        t_indices (list or int): Indices of time points to include
    """
    np.random.seed(random_seed)
    
    dataset = LatentEndpointDatasetAutodecodingMultimodal(
        healthy_hdf5_path=healthy_hdf5_path, # Pass hdf5_path here
        endpoint_name=endpoint_name,
        debug_limit=debug_limit,
        z_indices=z_indices,
        t_indices=t_indices,
        random_seed=random_seed
    )
    
    # Add logging
    logging.info(f"Dataset size before split: {len(dataset)}")
    logging.info(f"Number of cases: {sum(dataset.labels)}")
    logging.info(f"Number of controls: {len(dataset.labels) - sum(dataset.labels)}")
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check patient selection criteria.")
    
    # Separate indices by label
    case_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    control_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    # Calculate split sizes for each group
    n_train_cases = int(len(case_indices) * train_split)
    n_val_cases = int(len(case_indices) * val_split)
    n_test_cases = len(case_indices) - n_train_cases - n_val_cases
    
    n_train_controls = int(len(control_indices) * train_split)
    n_val_controls = int(len(control_indices) * val_split)
    n_test_controls = len(control_indices) - n_train_controls - n_val_controls
    
    # Shuffle indices
    np.random.shuffle(case_indices)
    np.random.shuffle(control_indices)
    
    # Split each group into train/val/test
    train_indices = (
        case_indices[:n_train_cases] +  # Training cases
        control_indices[:n_train_controls]  # Training controls
    )
    
    val_indices = (
        case_indices[n_train_cases:n_train_cases + n_val_cases] +  # Validation cases
        control_indices[n_train_controls:n_train_controls + n_val_controls]  # Validation controls
    )
    
    test_indices = (
        case_indices[n_train_cases + n_val_cases:] +  # Test cases
        control_indices[n_train_controls + n_val_controls:]  # Test controls
    )
    
    # Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Create dataloaders
    from torch.utils.data import SubsetRandomSampler
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    # Print detailed split information
    n_train_cases = sum(dataset.labels[i] == 1 for i in train_indices)
    n_train_controls = sum(dataset.labels[i] == 0 for i in train_indices)
    n_val_cases = sum(dataset.labels[i] == 1 for i in val_indices)
    n_val_controls = sum(dataset.labels[i] == 0 for i in val_indices)
    n_test_cases = sum(dataset.labels[i] == 1 for i in test_indices)
    n_test_controls = sum(dataset.labels[i] == 0 for i in test_indices)
    
    logging.info("Dataset split information:")
    logging.info(f"Train set: {len(train_indices)} total samples")
    logging.info(f"  - Cases: {n_train_cases}")
    logging.info(f"  - Controls: {n_train_controls}")
    logging.info(f"Validation set: {len(val_indices)} total samples")
    logging.info(f"  - Cases: {n_val_cases}")
    logging.info(f"  - Controls: {n_val_controls}")
    logging.info(f"Test set: {len(test_indices)} total samples")
    logging.info(f"  - Cases: {n_test_cases}")
    logging.info(f"  - Controls: {n_test_controls}")
    
    return train_loader, val_loader, test_loader

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

