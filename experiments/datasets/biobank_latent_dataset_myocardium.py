import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import h5py
import pandas as pd
import numpy as np
import os
import jax.numpy as jnp
import time
import traceback
from pathlib import Path
import nibabel as nib

        
class LatentEndpointDatasetEDESMiddleSlicesMyocardium(Dataset):
    def __init__(self, hdf5_path, endpoints_csv_path, endpoint_name='LVEF', 
                 cache_metadata=True, debug_limit=None, k_nearest=16, exclude_k_nearest=False):
        """
        Dataset combining latent representations from HDF5 with endpoints from CSV.
        
        Args:
            hdf5_path (str): Path to HDF5 file containing latent representations
            endpoints_csv_path (str): Path to CSV file with endpoints
            endpoint_name (str): Name of the endpoint column to use as target
            cache_metadata (bool): Whether to cache patient metadata
            debug_limit (int, optional): Limit number of patients for debugging
            k_nearest (int): Number of nearest latents to exclude when exclude_k_nearest is True
            exclude_k_nearest (bool): If True, use all latents except the k nearest ones
        """
        self.hdf5_path = hdf5_path
        self.endpoint_name = endpoint_name
        self.k_nearest = k_nearest
        self.exclude_k_nearest = exclude_k_nearest
        
        # Remove z_indices parameter as we'll calculate it per patient
        print(f"Initializing dataset with HDF5: {hdf5_path}")
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
        
        # Ensure patient ID column exists and convert to string for consistent comparison
        patient_id_col = next((col for col in self.endpoints_df.columns 
                               if col.lower() in ['f.eid']), None)
        if patient_id_col is None:
            raise ValueError("Could not find patient ID column in endpoints CSV")
            
        self.patient_id_col = patient_id_col
        self.endpoints_df[patient_id_col] = self.endpoints_df[patient_id_col].astype(str)
        
        # Filter out patients without the target endpoint or ED/ES timepoints
        self.endpoints_df = self.endpoints_df.dropna(subset=[endpoint_name, 'ED', 'ES'])
        print(f"After filtering for valid {endpoint_name} and ED/ES: {len(self.endpoints_df)} patients")
        
        # Get set of patients with valid endpoints for faster lookup
        patients_with_endpoints = set(self.endpoints_df[patient_id_col].values)
        print(f"Number of patients with valid endpoints: {len(patients_with_endpoints)}")
        
        # Open HDF5 file and get all available patient IDs
        try:
            with h5py.File(hdf5_path, 'r') as f:
                all_hdf5_patients = []
                patient_metadata = {}
                
                all_patients = list(f['patients'].keys())
                print(f"Total patients in HDF5: {len(all_patients)}")
                
                if debug_limit is not None and debug_limit > 0:
                    print(f"DEBUG MODE: Limiting to {debug_limit} patients")
                    all_patients = all_patients[:debug_limit]
                
                patients_to_check = [p for p in all_patients if p in patients_with_endpoints]
                print(f"Checking metadata for {len(patients_to_check)} patients with endpoints")
                
                for i, patient_id in enumerate(patients_to_check):
                    try:
                        patient_group = f['patients'][patient_id]
                        
                        z_dim = patient_group.attrs.get('Z', 0)
                        t_dim = patient_group.attrs.get('T', 0)
                        
                        # Calculate middle z-index for this patient
                        middle_z = z_dim // 2
                        
                        patient_metadata[patient_id] = {
                            'Z': z_dim,
                            'T': t_dim,
                            'H': patient_group.attrs.get('H', 0),
                            'W': patient_group.attrs.get('W', 0),
                            'middle_z': middle_z  # Store the middle z-index
                        }
                        
                        # Only include patients with at least 3 z-slices (to ensure there's a middle)
                        if z_dim >= 3:
                            all_hdf5_patients.append(patient_id)
                        else:
                            print(f"Patient {patient_id} has only {z_dim} z-slices, need at least 3")
                        
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

                # total_points = p_full.shape[1]  # Full shape is (1, num_points, 4)
                total_points = max(p_full.shape) # NOTE: Hacky way of making it work for the Autodecoding and the Meta-learning datasets
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
    
    def find_myocardium_center(self, segmentation):
        """
        Find the center of the myocardium (class 0.67) in the segmentation map
        Returns: (y, x) coordinates of the center
        """
        # Create binary mask for myocardium (class 0.67)
        myo_mask = (np.abs(segmentation - 0.67) < 0.01).astype(np.float32)
        
        # Find center of mass
        if np.any(myo_mask):
            y_indices, x_indices = np.nonzero(myo_mask)
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            return center_y, center_x
        return None

    def transform_coordinates(self, points, target_shape):
        """
        Transform coordinates from [-1, 1] range to image dimensions
        """
        # Take only the last two dimensions (height and width)
        points_2d = points[:, 2:]  # Shape: (N, 2)
        
        # Convert from [-1, 1] to [0, 1]
        points_2d = (points_2d + 1) / 2
        
        # Scale to image dimensions
        points_transformed = points_2d.copy()
        points_transformed[:, 0] = points_2d[:, 0] * (target_shape[0] - 1)  # height
        points_transformed[:, 1] = points_2d[:, 1] * (target_shape[1] - 1)  # width
        
        return points_transformed

    def find_latents_to_use(self, latent_coords, center_point):
        """
        Returns indices of latent points to use based on their distance from center point.
        If exclude_k_nearest is True, returns all indices except the k nearest points.
        Otherwise returns the k nearest points (original behavior).
        """
        distances = np.sqrt(
            (latent_coords[:, 0] - center_point[0])**2 + 
            (latent_coords[:, 1] - center_point[1])**2
        )
        
        if self.exclude_k_nearest:
            # Get all indices except the k nearest
            sorted_indices = np.argsort(distances)
            return sorted_indices[self.k_nearest:]  # All points except k nearest
        else:
            # Original behavior - get k nearest points
            return np.argsort(distances)[:self.k_nearest]
    
    def __getitem__(self, idx):
        try:
            patient_id = self.patient_ids[idx]
            
            # Get patient metadata
            patient_data = self.endpoints_df.loc[
                self.endpoints_df[self.patient_id_col] == patient_id
            ]
            endpoint_value = patient_data[self.endpoint_name].values[0]
            ed_timepoint = 0  # Always use t=0 for ED
            es_timepoint = int(patient_data['ES'].values[0])
            
            Z = self.patient_shapes[patient_id]['Z']
            T = self.patient_shapes[patient_id]['T']
            middle_z = self.patient_shapes[patient_id]['middle_z']
            
            if es_timepoint >= T:
                raise ValueError(f"Invalid ES timepoint for patient {patient_id}: ES={es_timepoint}, T={T}")
            
            # Load NIFTI segmentation data
            nifti_base_dir = "/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped"  # You might want to make this configurable
            patient_dir = Path(nifti_base_dir) / patient_id
            seg_path = patient_dir / "cropped_seg_sa.nii.gz"
            
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            
            # Load latent vectors
            with h5py.File(self.hdf5_path, 'r') as f:
                patient_group = f[f'patients/{patient_id}']
                
                try:
                    p_full = patient_group['p'][:].squeeze()
                    c_full = patient_group['c'][:].squeeze()
                    g_full = patient_group['g'][:].squeeze()
                except: 
                    p_full = patient_group['p'][:]
                    c_full = patient_group['c'][:]
                    g_full = patient_group['g'][:]
                
                # Calculate points per slice
                total_points = max(p_full.shape)
                num_latents = total_points // (Z * T)
                
                selected_p = []
                selected_c = []
                selected_g = []
                
                # Process ED and ES timepoints
                for t_idx in [ed_timepoint, es_timepoint]:
                    # Get slice indices
                    slice_start = (t_idx * Z + middle_z) * num_latents
                    slice_end = slice_start + num_latents
                
                    # Get slice data
                    p_slice = p_full[slice_start:slice_end]
                    c_slice = c_full[slice_start:slice_end]
                    g_slice = g_full[slice_start:slice_end]
                    seg_slice = seg_data[:, :, middle_z, t_idx]
                    
                    # Find myocardium center
                    center = self.find_myocardium_center(seg_slice)
                    if center is not None:
                        # Transform coordinates and find points to use
                        p_transformed = self.transform_coordinates(p_slice, seg_slice.shape)
                        selected_indices = self.find_latents_to_use(p_transformed, center)
                        
                        # Select points
                        selected_p.append(p_slice[selected_indices])
                        selected_c.append(c_slice[selected_indices])
                        selected_g.append(g_slice[selected_indices])
                
                # Concatenate ED and ES selections (outside the for loop)
                if len(selected_p) > 0:
                    p = jnp.array(np.concatenate(selected_p))
                    c = jnp.array(np.concatenate(selected_c))
                    g = jnp.array(np.concatenate(selected_g))
                    
                    return patient_id, (p, c, g), float(endpoint_value)
                else:
                    raise ValueError(f"No valid points found for patient {patient_id}")
                
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            print(traceback.format_exc())
            if idx > 0:
                return self.__getitem__(0)
            else:
                # Return empty arrays with correct shapes for k_nearest points
                empty_p = jnp.zeros((self.k_nearest * 2, 4))  # *2 for ED and ES
                empty_c = jnp.zeros((self.k_nearest * 2, 16))  # Adjust second dim based on your latent dim
                empty_g = jnp.zeros((self.k_nearest * 2, 1))
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
                      train_split=0.7, val_split=0.15,  # Modified split ratios
                      random_seed=42, num_workers=0, debug_limit=None, 
                      k_nearest=16, exclude_k_nearest=False):
    """
    Create train, validation, and test dataloaders with a 70/15/15 split.
    """
    try:
        full_dataset = LatentEndpointDatasetEDESMiddleSlicesMyocardium(
            hdf5_path, 
            endpoints_csv_path, 
            endpoint_name,
            debug_limit=debug_limit,
            k_nearest=k_nearest,
            exclude_k_nearest=exclude_k_nearest
        )
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        print(traceback.format_exc())
        raise
    
    np.random.seed(random_seed)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        print("WARNING: Dataset is empty!")
        return None, None, None
        
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    test_size = dataset_size - train_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create dataloaders
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False if num_workers == 0 else True
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
    
    print(f"Created dataloaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    
    return train_loader, val_loader, test_loader
