"""Dataset loading and synthetic data generation."""

import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from backend.config import DATASET_CACHE_DIR, DEFAULT_BATCH_SIZE, DEFAULT_TRAIN_SPLIT, XOR_RANDOM_SEED, POLYNOMIAL_RANDOM_SEED


def load_dataset(name: str, params: dict) -> dict:
    """Load or generate a dataset. Returns dict with loaders and metadata."""
    batch_size = params.get("batch_size", DEFAULT_BATCH_SIZE)

    if name == "mnist":
        return _load_mnist(batch_size, params)
    elif name == "cifar10":
        return _load_cifar10(batch_size, params)
    elif name == "cifar100":
        return _load_cifar100(batch_size, params)
    elif name == "xor":
        return _generate_xor(batch_size, params)
    elif name == "polynomial":
        return _generate_polynomial(batch_size, params)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_mnist(batch_size, params):
    normalize = params.get("normalize", True)
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)

    full_train = torchvision.datasets.MNIST(
        root=DATASET_CACHE_DIR, train=True, download=True, transform=transform)
    full_test = torchvision.datasets.MNIST(
        root=DATASET_CACHE_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_name": "MNIST",
        "input_shape": [1, 28, 28],
        "input_size": 784,
        "num_classes": 10,
        "train_samples": len(full_train),
        "test_samples": len(full_test),
        "task": "classification",
    }


def _load_cifar10(batch_size, params):
    normalize = params.get("normalize", True)
    transform_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if normalize:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transform_train.append(norm)
        transform_test.append(norm)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    full_train = torchvision.datasets.CIFAR10(
        root=DATASET_CACHE_DIR, train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(
        root=DATASET_CACHE_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_name": "CIFAR-10",
        "input_shape": [3, 32, 32],
        "input_size": 3072,
        "num_classes": 10,
        "train_samples": len(full_train),
        "test_samples": len(full_test),
        "task": "classification",
    }


def _load_cifar100(batch_size, params):
    normalize = params.get("normalize", True)
    transform_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if normalize:
        norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform_train.append(norm)
        transform_test.append(norm)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    full_train = torchvision.datasets.CIFAR100(
        root=DATASET_CACHE_DIR, train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR100(
        root=DATASET_CACHE_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_name": "CIFAR-100",
        "input_shape": [3, 32, 32],
        "input_size": 3072,
        "num_classes": 100,
        "train_samples": len(full_train),
        "test_samples": len(full_test),
        "task": "classification",
    }


def _generate_xor(batch_size, params):
    """Generate XOR classification data.
    Four clusters at (0,0), (0,1), (1,0), (1,1).
    Label 0: both same (0,0), (1,1). Label 1: different (0,1), (1,0).
    """
    num_samples = params.get("num_samples", 1000)
    noise = params.get("noise_level", 0.05)

    np.random.seed(XOR_RANDOM_SEED)
    centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    labels_center = np.array([0, 1, 1, 0])

    indices = np.random.randint(0, 4, num_samples)
    data = centers[indices] + np.random.randn(num_samples, 2) * noise
    labels = labels_center[indices]

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(data, labels)
    split = int(DEFAULT_TRAIN_SPLIT * num_samples)
    train_ds = torch.utils.data.Subset(ds, range(split))
    test_ds = torch.utils.data.Subset(ds, range(split, num_samples))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_name": "XOR",
        "input_shape": [2],
        "input_size": 2,
        "num_classes": 2,
        "train_samples": split,
        "test_samples": num_samples - split,
        "task": "classification",
    }


def _generate_polynomial(batch_size, params):
    """Generate polynomial regression data.
    y = w_0 + w_1*x + w_2*x^2 + ... + w_d*x^d + noise
    """
    num_samples = params.get("num_samples", 1000)
    degree = params.get("degree", 3)
    noise = params.get("noise_level", 0.1)
    input_dim = params.get("input_dim", 1)

    np.random.seed(POLYNOMIAL_RANDOM_SEED)
    x = np.random.uniform(-2, 2, (num_samples, input_dim)).astype(np.float32)

    # Generate random polynomial coefficients, sum over input dims
    y = np.zeros(num_samples, dtype=np.float32)
    for d in range(degree + 1):
        coeffs = np.random.randn(input_dim).astype(np.float32) * (0.5 / (d + 1))
        y += np.sum(coeffs * (x ** d), axis=1)

    y += np.random.randn(num_samples).astype(np.float32) * noise

    x_t = torch.tensor(x)
    y_t = torch.tensor(y).unsqueeze(1)

    ds = torch.utils.data.TensorDataset(x_t, y_t)
    split = int(DEFAULT_TRAIN_SPLIT * num_samples)
    train_ds = torch.utils.data.Subset(ds, range(split))
    test_ds = torch.utils.data.Subset(ds, range(split, num_samples))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "dataset_name": "Polynomial",
        "input_shape": [input_dim],
        "input_size": input_dim,
        "num_classes": 1,
        "train_samples": split,
        "test_samples": num_samples - split,
        "task": "regression",
    }
