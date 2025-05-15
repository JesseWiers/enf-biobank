import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np
import os
import jax.numpy as jnp
import time
import traceback

class LatentEndpointDataset(Dataset):
    def __init__(self, hdf5_path, endpoints_csv_path, endpoint_name='LVEF', 
                 z_indices=[0, 2, 4], t_indices=[0, 10, 20, 30, 40], 
                 cache_metadata=True, debug_limit=None):
        """
        Dataset combining latent representations from HDF5 with endpoints from CSV,
        filtering specific z and t slices.
        
        Args:
            hdf5_path (str): Path to HDF5 file containing latent representations
            endpoints_csv_path (str): Path to CSV file with endpoints
            endpoint_name (str): Name of the endpoint column to use as target
            z_indices (list): Indices of z-slices to include (e.g., [0, 2, 4])
            t_indices (list): Indices of time points to include (e.g., [0, 10, 20, 30, 40])
            cache_metadata (bool): Whether to cache patient metadata
            debug_limit (int, optional): Limit number of patients for debugging
        """
        self.hdf5_path = hdf5_path
        self.endpoint_name = endpoint_name
        self.z_indices = sorted(z_indices)  # Ensure indices are sorted
        self.t_indices = sorted(t_indices)  # Ensure indices are sorted
        
        print(f"Initializing dataset with HDF5: {hdf5_path}")
        print(f"Endpoints CSV: {endpoints_csv_path}")
        print(f"Using z-indices: {self.z_indices}, t-indices: {self.t_indices}")
        
        start_time = time.time()
        
        # Load endpoints CSV
        try:
            self.endpoints_df = pd.read_csv(endpoints_csv_path)
            print(f"Loaded endpoints CSV with {len(self.endpoints_df)} rows")
        except Exception as e:
            print(f"Error loading endpoints CSV: {str(e)}")
            raise
        
        # Ensure patient ID column exists and convert to string for consistent comparison
        patient_id_col = next((col for col in self.endpoints_df.columns 
                               if col.lower() in ['f.eid']), None)
        if patient_id_col is None:
            raise ValueError("Could not find patient ID column in endpoints CSV")
            
        self.patient_id_col = patient_id_col
        self.endpoints_df[patient_id_col] = self.endpoints_df[patient_id_col].astype(str)
        
        # Filter out patients without the target endpoint
        self.endpoints_df = self.endpoints_df.dropna(subset=[endpoint_name])
        print(f"After filtering for valid {endpoint_name}: {len(self.endpoints_df)} patients")
        
        # Get set of patients with valid endpoints for faster lookup
        patients_with_endpoints = set(self.endpoints_df[patient_id_col].values)
        print(f"Number of patients with valid endpoints: {len(patients_with_endpoints)}")
        
        # Open HDF5 file and get all available patient IDs
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Check all patients to ensure they have enough z-slices and t-slices
                all_hdf5_patients = []
                patient_metadata = {}
                
                # Get all patient IDs first
                all_patients = list(f['patients'].keys())
                print(f"Total patients in HDF5: {len(all_patients)}")
                
                # Apply debug limit if specified
                if debug_limit is not None and debug_limit > 0:
                    print(f"DEBUG MODE: Limiting to {debug_limit} patients")
                    all_patients = all_patients[:debug_limit]
                
                # Only process patients that have endpoints (filter first)
                patients_to_check = [p for p in all_patients if p in patients_with_endpoints]
                print(f"Checking metadata for {len(patients_to_check)} patients with endpoints")
                
                # Process patients in smaller batches
                for i, patient_id in enumerate(patients_to_check):
                    try:
                        patient_group = f['patients'][patient_id]
                        
                        # Extract Z and T dimensions from attributes
                        z_dim = patient_group.attrs.get('Z', 0)
                        t_dim = patient_group.attrs.get('T', 0)
                        
                        # Store metadata regardless
                        patient_metadata[patient_id] = {
                            'Z': z_dim,
                            'T': t_dim,
                            'H': patient_group.attrs.get('H', 0),
                            'W': patient_group.attrs.get('W', 0),
                        }
                        
                        # Only include patients with enough z and t slices
                        if z_dim > max(self.z_indices) and t_dim > max(self.t_indices):
                            all_hdf5_patients.append(patient_id)
                        else:
                            if z_dim <= max(self.z_indices):
                                print(f"Patient {patient_id} has only {z_dim} z-slices, need at least {max(self.z_indices)+1}")
                            if t_dim <= max(self.t_indices):
                                print(f"Patient {patient_id} has only {t_dim} t-slices, need at least {max(self.t_indices)+1}")
                        
                        # Show progress periodically
                        if (i+1) % 50 == 0:
                            print(f"Processed {i+1}/{len(patients_to_check)} patients...")
                            
                    except Exception as e:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        continue
                        
                print(f"Found {len(all_hdf5_patients)} patients with sufficient slices")
                
                # Store metadata for later use
                self.patient_metadata = patient_metadata
        except Exception as e:
            print(f"Error accessing HDF5 file: {str(e)}")
            print(traceback.format_exc())
            raise
            
        # Find patients that are in both the HDF5 file and have valid endpoints
        valid_patients = set(patients_with_endpoints) & set(all_hdf5_patients)
        print(f"Valid patients (with endpoints and sufficient slices): {len(valid_patients)}")
        
        # Filter endpoints dataframe to only include valid patients
        self.endpoints_df = self.endpoints_df[self.endpoints_df[patient_id_col].isin(valid_patients)]
        
        # Create a list of patient IDs in the order we'll use them
        self.patient_ids = self.endpoints_df[patient_id_col].tolist()
        
        # Store shapes for quick access
        self.patient_shapes = {pid: self.patient_metadata[pid] for pid in self.patient_ids}
        
        # Pre-calculate number of points per slice
        # Check the first patient to determine this
        if len(self.patient_ids) > 0:
            with h5py.File(hdf5_path, 'r') as f:
                first_patient = self.patient_ids[0]
                first_z = self.patient_shapes[first_patient]['Z']
                first_t = self.patient_shapes[first_patient]['T']
                 
                # Calculate points per slice from total points
                p_full = f[f'patients/{first_patient}/p'][:]
                total_points = p_full.shape[1]  # Full shape is (1, num_points, 4)
                self.points_per_slice = total_points // (first_z * first_t)
                print(f"Each patient has {self.points_per_slice} points per slice")
        else:
            self.points_per_slice = 0
            print("WARNING: No valid patients found!")
        
        end_time = time.time()
        print(f"Dataset initialization completed in {end_time - start_time:.2f} seconds")
        print(f"Dataset contains {len(self.patient_ids)} patients")
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            
            # Get endpoint value
            endpoint_value = self.endpoints_df.loc[
                self.endpoints_df[self.patient_id_col] == patient_id, 
                self.endpoint_name
            ].values[0]
            
            # Get shape information for this patient
            Z = self.patient_shapes[patient_id]['Z']
            T = self.patient_shapes[patient_id]['T']
            
            # Calculate points per slice if we don't have it
            if not hasattr(self, 'points_per_slice') or self.points_per_slice == 0:
                with h5py.File(self.hdf5_path, 'r') as f:
                    p_full = f[f'patients/{patient_id}/p'][:]
                    total_points = p_full.shape[1]
                    self.points_per_slice = total_points // (Z * T)
            
            num_latents = self.points_per_slice
            
            # Load latent representation from HDF5
            with h5py.File(self.hdf5_path, 'r') as f:
                p_full = f[f'patients/{patient_id}/p'][:]  # Shape: (1, num_points, 4)
                c_full = f[f'patients/{patient_id}/c'][:]  # Shape: (1, num_points, latent_dim)
                g_full = f[f'patients/{patient_id}/g'][:]  # Shape: (1, num_points, 1)
            
            # Remove batch dimension
            p_full = p_full.squeeze(0)  # Now shape: (num_points, 4)
            c_full = c_full.squeeze(0)  # Now shape: (num_points, latent_dim)
            g_full = g_full.squeeze(0)  # Now shape: (num_points, 1)
            
            # Initialize filtered arrays
            selected_indices = []
            
            # For each selected t-value
            for t_idx in self.t_indices:
                if t_idx < T:  # Make sure the t-index is valid
                    # For each selected z-index
                    for z_idx in self.z_indices:
                        if z_idx < Z:  # Make sure the z-index is valid
                            # Calculate start index for this slice
                            start_idx = (t_idx * Z + z_idx) * num_latents
                            end_idx = start_idx + num_latents
                            
                            # Verify bounds
                            if end_idx <= p_full.shape[0]:
                                # Add indices for this slice
                                selected_indices.extend(range(start_idx, end_idx))
            
            if not selected_indices:
                raise ValueError(f"No valid indices selected for patient {patient_id}")
                
            # Filter the points
            try:
                p = p_full[selected_indices]
                c = c_full[selected_indices]
                g = g_full[selected_indices]
                
                # Convert to JAX arrays
                p = jnp.array(p)
                c = jnp.array(c)
                g = jnp.array(g)
                
                # Create the latent tuple z
                z = (p, c, g)
                
                # Create a float value for the endpoint
                endpoint_value = float(endpoint_value)
                
                # Return patient_id and the latent tuple z
                return patient_id, z, endpoint_value
                
            except Exception as e:
                print(f"Error filtering points for patient {patient_id}: {str(e)}")
                print(f"Selected indices: {len(selected_indices)}, p_full shape: {p_full.shape}")
                raise
                
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            print(traceback.format_exc())
            # Return a placeholder value rather than crashing
            if idx > 0:
                return self.__getitem__(0)  # Try returning the first item instead
            else:
                # Create empty placeholder data with appropriate shapes
                empty_p = jnp.zeros((10, 4))
                empty_c = jnp.zeros((10, 16))  # Assuming latent_dim=16
                empty_g = jnp.zeros((10, 1))
                return "error", (empty_p, empty_c, empty_g), 0.0

# Custom collate function for JAX arrays
def collate_fn(batch):
    try:
        # Unpack the batch
        patient_ids, z_tuples, endpoints = zip(*batch)
        
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
        
        # Convert endpoints to array
        endpoints_array = jnp.array(endpoints)
        
        return patient_ids, z_batched, endpoints_array
    except Exception as e:
        print(f"Error in collate_fn: {str(e)}")
        print(f"Batch type: {type(batch)}, len: {len(batch)}")
        print(f"First batch item: {batch[0]}")
        # Return an empty batch as fallback
        return [], (jnp.array([]), jnp.array([]), jnp.array([])), jnp.array([])

def create_dataloaders(hdf5_path, endpoints_csv_path, batch_size=16, endpoint_name='LVEF', 
                      z_indices=[0, 2, 4], t_indices=[0, 10, 20, 30, 40], 
                      val_split=0.2, test_split=0.1, random_seed=42, 
                      num_workers=0, debug_limit=None):  # Set num_workers=0 for debugging
    """
    Create train, validation, and test dataloaders with standardized z and t slices.
    """
    # Create dataset
    try:
        full_dataset = LatentEndpointDataset(
            hdf5_path, 
            endpoints_csv_path, 
            endpoint_name,
            z_indices=z_indices,
            t_indices=t_indices,
            debug_limit=debug_limit
        )
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        print(traceback.format_exc())
        raise
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        print("WARNING: Dataset is empty!")
        return None, None, None
        
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size - test_size
    
    # Ensure at least one sample in each split
    if train_size < 1 or val_size < 1 or test_size < 1:
        print("WARNING: Not enough data for splits, using single-sample splits")
        train_size = max(1, dataset_size - 2)
        val_size = 1
        test_size = min(1, dataset_size - train_size - val_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create dataloaders with custom samplers and collate function
    from torch.utils.data import SubsetRandomSampler
    
    # Start with fewer workers for debugging
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,  # Start with 0 workers for debugging
        collate_fn=collate_fn,
        pin_memory=False,  # Disable pin_memory with JAX
        persistent_workers=False if num_workers == 0 else True  # Only persistent if using workers
    )
    
    val_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
    )
    
    # Calculate resulting point cloud size
    if hasattr(full_dataset, 'points_per_slice') and full_dataset.points_per_slice > 0:
        valid_z_indices = [z for z in z_indices if z < max(full_dataset.patient_shapes[pid]['Z'] for pid in full_dataset.patient_ids)]
        valid_t_indices = [t for t in t_indices if t < max(full_dataset.patient_shapes[pid]['T'] for pid in full_dataset.patient_ids)]
        
        points_per_patient = len(valid_z_indices) * len(valid_t_indices) * full_dataset.points_per_slice
    else:
        points_per_patient = "unknown"
    
    print(f"Created dataloaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    print(f"Each patient has approximately {points_per_patient} points")
    print(f"Using {num_workers} worker processes")
    
    return train_loader, val_loader, test_loader

# Usage example with debugging options:
if __name__ == "__main__":
    hdf5_path = "/projects/prjs1252/data_jesse/latent_dataset_4d.h5"
    endpoints_csv_path = "/projects/prjs1252/data_jesse/metadata/filtered_endpoints.csv"
    
    # Start with a debug version - no workers, limited patients
    train_loader, val_loader, test_loader = create_dataloaders(
        hdf5_path=hdf5_path,
        endpoints_csv_path=endpoints_csv_path,
        batch_size=1,  # Small batch size for debugging
        endpoint_name='LVEF',
        z_indices=[0, 2, 4],  
        t_indices=[0, 10, 20, 30, 40],
        num_workers=0,  # No multiprocessing for debugging
        debug_limit=10  # Only use 10 patients for debugging
    )
    
    # Test the first batch
    print("Testing first batch from train_loader...")
    try:
        batch = next(iter(train_loader))
        
        # Unpack the batch
        patient_ids, z_tuples, endpoints = batch
        
        # Access the first patient's data
        patient_id = patient_ids[0]
        z = z_tuples[0]  # This is a tuple of (p, c, g)
        endpoint = endpoints[0]
        
        # Unpack z tuple
        p, c, g = z
        
        print(f"Patient {patient_id}:")
        print(f"- p shape: {p.shape}, type: {type(p)}")
        print(f"- c shape: {c.shape}, type: {type(c)}")
        print(f"- g shape: {g.shape}, type: {type(g)}")
        print(f"- LVEF: {endpoint}")
        
        print("First batch test successful!")
    except Exception as e:
        print(f"Error testing first batch: {str(e)}")
        print(traceback.format_exc())
    
    # After debugging is successful, you can increase workers and remove limits
