#!/usr/bin/env python
"""
data.py - Data loading and preprocessing module.
This module now supports MNIST, FashionMNIST, EMNIST (using the 'balanced' split),
KMNIST, and QMNIST.
Do NOT alter any of the original code logic.
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import rawarrview, reshape_image_batch

# Create a transform chain that:
# 1. Converts a PIL image to a tensor with values in [0,1].
# 2. Binarizes the image by sampling uniform noise and thresholding.
# 3. Permutes the tensor to [H, W, C] for visualization.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (torch.rand_like(x) < x).float()),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])

# Define a simple wrapper to optionally "uncondition" the labels.
# This class works for any underlying dataset returning (image, label).
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset, unconditional=False, p=0.2):
        self.base_dataset = base_dataset
        self.unconditional = unconditional
        self.p = p
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Adjust label as in the original code (label + 1)
        label = label + 1
        if self.unconditional and torch.rand(1).item() < self.p:
            label = 0
        return image, label

def load_data(args):
    """
    Downloads and prepares the specified dataset.
    Supported options: MNIST, FashionMNIST, EMNIST (balanced split), KMNIST, and QMNIST.
    Uses command line arguments: --data_root, --batch_size, and --dataset.
    """
    dataset_name = args.dataset.lower()
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
    elif dataset_name == "fashionmnist":
        train_dataset = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)
    elif dataset_name == "emnist":
        # Using the 'balanced' split for EMNIST.
        train_dataset = datasets.EMNIST(root=args.data_root, split='balanced', train=True, download=True, transform=transform)
        test_dataset  = datasets.EMNIST(root=args.data_root, split='balanced', train=False, download=True, transform=transform)
    elif dataset_name == "kmnist":
        train_dataset = datasets.KMNIST(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset  = datasets.KMNIST(root=args.data_root, train=False, download=True, transform=transform)
    elif dataset_name == "qmnist":
        train_dataset = datasets.QMNIST(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset  = datasets.QMNIST(root=args.data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Choose from MNIST, FashionMNIST, EMNIST, KMNIST, or QMNIST.")

    # Wrap the datasets to optionally apply label modifications.
    train_dataset = CustomMNIST(train_dataset, unconditional=True, p=0.2)
    test_dataset  = CustomMNIST(test_dataset, unconditional=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # (Optional visualization code is commented out.)
    """
    images, labels = next(iter(train_loader))
    grid = reshape_image_batch(images.squeeze(-1).numpy())
    rawarrview(grid, cmap='bone_r')
    """
    return train_loader, test_loader
