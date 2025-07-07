# MNIST/dataloader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def get_mnist_loaders(batch_size=32, num_train=8000):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),                      # downsample to 256 pixels
        transforms.ToTensor(),                          # convert to tensor [C,H,W]
        transforms.Lambda(lambda x: x.view(-1)),        # flatten to 1D [16]
        transforms.Lambda(lambda x: x * torch.pi)         # scale [0,1] → [0,π]
    ])
    
    train_set_full = datasets.MNIST(root='./MNIST/data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./MNIST/data', train=False, download=True, transform=transform)
    
    # Check bounds
    num_train = min(num_train, len(train_set_full))
    # Random subset of training data
    subset_indices = random.sample(range(len(train_set_full)), num_train)
    train_subset = Subset(train_set_full, subset_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, test_loader
