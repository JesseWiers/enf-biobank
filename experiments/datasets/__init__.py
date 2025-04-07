import torch
import torchvision
from PIL import Image
import numpy as np

from experiments.datasets.ombria_dataset import Ombria
from torchvision.datasets import CIFAR10
from experiments.datasets.biobank_dataset import BiobankNifti, BiobankNifti3D


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


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int, num_train: int, num_test: int, seed: int, z_indices: list[int]):
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
    elif dataset_name == "3d_biobank":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankNifti3D(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="train", transform=transforms, num_patients_train=num_train, z_indices=z_indices)
        test_dset = BiobankNifti3D(root='/home/jwiers/deeprisk/ukbb_cardiac/datasets/n=1200', split="test", transform=transforms, num_patients_test=num_test, z_indices=z_indices)
    else: 
        raise NotImplementedError("Moving MNIST dataset not implemented yet.")
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=True,
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

