import torch
import torchvision
from PIL import Image
import numpy as np

from experiments.datasets.ombria_dataset import Ombria
from torchvision.datasets import CIFAR10
from experiments.datasets.biobank_dataset import BiobankNifti, BiobankNifti3D, BiobankNiftiV2, BiobankNifti3DV2, BiobankNiftiLVEF, BiobankNiftiLVEF3D, BiobankNiftiLVEF4D
from experiments.datasets.biobank_dataset_endpoint_images import ImageEndpointDataset
from experiments.datasets.autodecoding.biobank_multimodal_endpoint_dataset import EndpointDatasetMultiModal
from experiments.datasets.autodecoding.biobank_endpoint_dataset import EndpointDataset
from experiments.datasets.autodecoding.biobank_endpoint_dataset_specific import EndpointDatasetSpecific

def image_to_numpy(image: Image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image) / 255


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


def  get_dataloaders(dataset_name: str, batch_size: int, num_workers: int, num_train: int, num_test: int, seed: int, z_indices:list[int] = [0,1], t_indices:list[int] = [0,1],  shuffle_train: bool = True, 
                     mosaic_augment: bool = False, num_patients: int = None, endpoint_name: str = None):
    """ 
    Returns specified dataset dataloaders.

    Args:
        dataset: The dataset to use. Can be 'ombria'.
        batch_size: The batch size to use.
        num_workers: The number of workers to use for the DataLoader.
    """    
    
    # Create generator with seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "ombria":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = Ombria(root="./data/ombria", split="train", transform=transforms, download=True)
        test_dset = Ombria(root="./data/ombria", split="test", transform=transforms, download=True)
    elif dataset_name == "cifar10":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = CIFAR10(root="./data/cifar10", train=True, transform=transforms, download=True)
        test_dset = CIFAR10(root="./data/cifar10", train=False, transform=transforms, download=True)
    elif dataset_name == "2d_biobank":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNifti(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="train", transform=transforms, num_patients_train=num_train)
        test_dset = BiobankNifti(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="test", transform=transforms, num_patients_test=num_test)
    elif dataset_name == "2d_biobank_v2":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNiftiV2(root='/projects/prjs1252/data_jesse/cmr_cropped', split="train", transform=transforms, num_patients_train=num_train, mosaic_augment=True)
        test_dset = BiobankNiftiV2(root='/projects/prjs1252/data_jesse/cmr_cropped', split="test", transform=transforms, num_patients_test=num_test, mosaic_augment=True)
    elif dataset_name == "biobank_lvef":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNiftiLVEF(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="train", transform=transforms, num_patients_train=num_train)
        test_dset = BiobankNiftiLVEF(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="test", transform=transforms, num_patients_test=num_test)
    elif dataset_name == "biobank_lvef_3d":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNiftiLVEF3D(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices)
        test_dset = BiobankNiftiLVEF3D(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices)
    elif dataset_name == "biobank_lvef_4d":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNiftiLVEF4D(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices, t_indices=t_indices)
        test_dset = BiobankNiftiLVEF4D(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices, t_indices=t_indices)
    elif dataset_name == "3d_biobank":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNifti3D(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices)
        test_dset = BiobankNifti3D(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices)
    elif dataset_name == "3d_biobank_v2":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNifti3DV2(root='/projects/prjs1252/data_jesse/cmr_cropped', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices)
        test_dset = BiobankNifti3DV2(root='/projects/prjs1252/data_jesse/cmr_cropped', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices)
    elif dataset_name == "multi_modal":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = EndpointDatasetMultiModal(root='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', ecg_path='/projects/prjs1252/data_jesse_final_v3/ECGs_median_leads.pth', num_patients=num_patients, z_indices=z_indices, t_indices=t_indices)
        test_dset = EndpointDatasetMultiModal(root='/projects/prjs1252/data_jesse_final_v3/nifti_dataset', ecg_path='/projects/prjs1252/data_jesse_final_v3/ECGs_median_leads.pth', num_patients=num_patients, z_indices=z_indices, t_indices=t_indices)
    elif dataset_name == "endpoints_4d":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = EndpointDataset(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices, t_indices=t_indices)
        test_dset = EndpointDataset(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices, t_indices=t_indices)
    elif dataset_name == "endpoints_4d_specific":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = EndpointDatasetSpecific(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', endpoint_name=endpoint_name, transform=transforms, num_patients_train=num_train, z_indices=z_indices, t_indices=t_indices)
        test_dset = EndpointDatasetSpecific(root='/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped', endpoint_name=endpoint_name, transform=transforms, num_patients_test=num_test, z_indices=z_indices, t_indices=t_indices)
    else: 
        raise NotImplementedError("Dataset not implemented yet.")
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,  # Add generator for reproducible shuffling
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)  # Seed workers
    )
    
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,  # Add generator for reproducible shuffling
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)  # Seed workers
    )

    return train_loader, test_loader

