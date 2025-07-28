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

class LatentEndpointDataset(Dataset):
    def __init__(self, hdf5_path, endpoint_name, endpoints_dir="/projects/prjs1252/data_jesse_final_v3/endpoints/",
                 z_indices=-1, t_indices=-1, debug_limit=None, random_seed=42):
        """
        Dataset combining latent representations with balanced endpoint cases and controls.
        
        Args:
            hdf5_path (str): Path to HDF5 file containing latent representations
            endpoint_name (str): Name of the endpoint (e.g., 'sudden_cardiac_death')
            endpoints_dir (str): Directory containing endpoint CSV files
            z_indices (list or int): Indices of z-slices to include (e.g., [0, 2, 4]) or -1 for all slices
            t_indices (list or int): Indices of time points to include (e.g., [0, 10, 20, 30, 40]) or -1 for all timepoints
            debug_limit (int, optional): Limit number of patients for debugging
        """
        np.random.seed(random_seed)
        self.hdf5_path = hdf5_path
        
        # Load case and control EIDs
        endpoint_path = os.path.join(endpoints_dir, f"{endpoint_name}_eids.csv")
        healthy_path = os.path.join(endpoints_dir, "healthy_matched_final_eids.csv")
        
        # First process cases
        cases_df = pd.read_csv(endpoint_path)
        case_eid_col = next(col for col in cases_df.columns if col.lower() in ['f.eid', 'eid'])
        cases = set(cases_df[case_eid_col].astype(str))
        
        with h5py.File(hdf5_path, 'r') as f:
            # Check Z dimension for cases first
            available_cases = []
            excluded_cases = 0
            
            for patient_id in tqdm(cases, desc="Filtering cases by Z dimension"):
                if patient_id not in f['patients']:
                    logging.info(f"Case patient {patient_id} not found in HDF5 file")
                    continue
                    
                Z = f[f'patients/{patient_id}'].attrs['Z']
                if Z >= 5:
                    available_cases.append(patient_id)
                else:
                    logging.info(f"Excluded case patient {patient_id} with Z={Z}")
                    excluded_cases += 1
            
            n_cases = len(available_cases)
            logging.info(f"Found {n_cases} valid cases, excluded {excluded_cases} cases with fewer than 5 Z slices")
            
            if n_cases == 0:
                raise ValueError("No valid cases found in HDF5 file")
            
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
            
            for patient_id in tqdm(controls_to_check, desc="Filtering controls by Z dimension"):
                if len(available_controls) >= n_cases:
                    break  # Stop once we have enough controls
                    
                if patient_id not in f['patients']:
                    logging.info(f"Control patient {patient_id} not found in HDF5 file")
                    continue
                    
                Z = f[f'patients/{patient_id}'].attrs['Z']
                if Z >= 5:
                    available_controls.append(patient_id)
                else:
                    logging.info(f"Excluded control patient {patient_id} with Z={Z}")
                    excluded_controls += 1
            
            # If we don't have enough controls, we might need to sample more
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
                            
                        if patient_id not in f['patients']:
                            continue
                            
                        Z = f[f'patients/{patient_id}'].attrs['Z']
                        if Z >= 5:
                            available_controls.append(patient_id)
                        else:
                            excluded_controls += 1
                    
                    remaining_controls -= additional_to_check
            
            # Take only the number of controls we need
            available_controls = available_controls[:n_cases]
            
            logging.info(f"Found {len(available_controls)} valid controls, excluded {excluded_controls} controls with fewer than 5 Z slices")
            
            # Combine cases and controls
            self.patient_ids = available_cases + available_controls
            self.labels = [1] * len(available_cases) + [0] * len(available_controls)
            
            # Get dimensions from first patient
            first_patient = self.patient_ids[0]
            patient_group = f['patients'][first_patient]
            z_dim = patient_group.attrs['Z']  
            t_dim = patient_group.attrs['T']
            
            # Process indices
            self.z_indices = z_indices
            
            if t_indices == -1:
                self.t_indices = list(range(t_dim))
            else:
                self.t_indices = t_indices if isinstance(t_indices, list) else list(t_indices)
            
            # Get points per slice for a single patient
            p = patient_group['p'][:]
            self.points_per_slice = p.shape[1] // (z_dim * t_dim)

            logging.info(f"Dataset initialized with {len(self.patient_ids)} patients ({len(available_cases)} cases + {len(available_controls)} controls)")
            # logging.info(f"Using {len(self.z_indices)} Z-slices and {len(self.t_indices)} timepoints")

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            patient_group = f[f'patients/{patient_id}']
            try: 
                p_full = f[f'patients/{patient_id}/p'][:].squeeze(0)
                c_full = f[f'patients/{patient_id}/c'][:].squeeze(0)
                g_full = f[f'patients/{patient_id}/g'][:].squeeze(0)
            except: 
                p_full = f[f'patients/{patient_id}/p'][:]
                c_full = f[f'patients/{patient_id}/c'][:]
                g_full = f[f'patients/{patient_id}/g'][:]
            
            # Get dimensions for this patient
            T = patient_group.attrs['T']
            Z = patient_group.attrs['Z']
            
            # Get z-indices based on this patient's Z dimension
            if isinstance(self.z_indices, int):
                if self.z_indices == -1:
                    z_indices_to_use = list(range(Z))
                else:
                    z_indices_to_use = get_middle_indices(Z, self.z_indices)
            else:
                z_indices_to_use = self.z_indices
                
            # logging.info(f"Patient ID: {patient_id}")
            # logging.info(f"Z={Z}")
            # logging.info(f"Z indices to use: {z_indices_to_use}")
            
            selected_indices = []
            
            # Get the number of points
            num_points = p_full.shape[0] 
            
            # For each selected t-value
            for t_idx in self.t_indices:
                # For each selected z-index
                for z_idx in z_indices_to_use:
                    if z_idx >= Z:
                        logging.warning(f"Skipping z_idx {z_idx} for patient {patient_id} with Z={Z}")
                        continue
                        
                    # Calculate start index for this slice
                    start_idx = (t_idx * Z + z_idx) * self.points_per_slice
                    end_idx = start_idx + self.points_per_slice
                    
                    # Add indices for this slice if within bounds
                    if end_idx <= num_points:
                        selected_indices.extend(range(start_idx, end_idx))
            
            if not selected_indices:
                logging.error(f"Patient {patient_id} dimensions: T={T}, Z={Z}")
                logging.error(f"Selected t_indices: {self.t_indices}")
                logging.error(f"Selected z_indices: {z_indices_to_use}")
                logging.error(f"Points per slice: {self.points_per_slice}")
                logging.error(f"p_full shape: {p_full.shape}")
                logging.error(f"Number of points: {num_points}")
                raise ValueError(f"No valid indices selected for patient {patient_id}")
                
            # Convert selected_indices to JAX array and use proper indexing
            selected_indices = jnp.array(selected_indices)
            
            # Handle different possible shapes of input arrays
            if p_full.ndim == 2:
                p = jnp.array(p_full[selected_indices])
            else:
                p = jnp.array(p_full[:, selected_indices].squeeze(0))
                
            if c_full.ndim == 2:
                c = jnp.array(c_full[selected_indices])
            else:
                c = jnp.array(c_full[:, selected_indices].squeeze(0))
                
            if g_full.ndim == 2:
                g = jnp.array(g_full[selected_indices])
            else:
                g = jnp.array(g_full[:, selected_indices].squeeze(0))
            
            z = (p, c, g)
            
            return patient_id, z, label


# Custom collate function for JAX arrays
def collate_fn(batch):
    try:
        # Unpack the batch
        patient_ids, z_tuples, labels = zip(*batch)
        
        # Separate p, c, g from z_tuples
        p_list = [z[0] for z in z_tuples]
        c_list = [z[1] for z in z_tuples]
        g_list = [z[2] for z in z_tuples]
        
        # Stack each component to add batch dimension
        p_batched = jnp.stack(p_list)  # Shape: (B, num_points, 4)
        c_batched = jnp.stack(c_list)  # Shape: (B, num_points, latent_dim)
        g_batched = jnp.stack(g_list)  # Shape: (B, num_points, 1)
        
        # Create batched z_tuple
        z_batched = (p_batched, c_batched, g_batched)
        
        # Convert labels to array
        labels_array = jnp.array(labels)
        
        return patient_ids, z_batched, labels_array
    except Exception as e:
        logging.error(f"Error in collate_fn: {str(e)}")
        logging.error(f"Batch type: {type(batch)}, len: {len(batch)}")
        logging.error(f"First batch item: {batch[0]}")
        # Return an empty batch as fallback
        return [], (jnp.array([]), jnp.array([]), jnp.array([])), jnp.array([])

def create_dataloaders(hdf5_path, endpoint_name, batch_size=16, 
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
    
    dataset = LatentEndpointDataset(
        hdf5_path=hdf5_path,
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

