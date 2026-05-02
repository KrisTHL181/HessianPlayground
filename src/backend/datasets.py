"""Dataset loading and synthetic data generation."""

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from backend.config import (
    DATASET_CACHE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRAIN_SPLIT,
    POLYNOMIAL_RANDOM_SEED,
    XOR_RANDOM_SEED,
)


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
    elif name == "fashion_mnist":
        return _load_fashion_mnist(batch_size, params)
    elif name == "synthetic_regression":
        return _generate_synthetic_regression(batch_size, params)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_torchvision_dataset(dataset_cls, dataset_kwargs, batch_size,
                               transform_train, transform_test,
                               dataset_name, input_shape, num_classes):
    """Load a torchvision dataset with train/test split and DataLoaders."""
    normalize = dataset_kwargs.pop("normalize", True)
    if not normalize:
        transform_train = transforms.Compose([t for t in transform_train.transforms if not isinstance(t, transforms.Normalize)])
        transform_test = transforms.Compose([t for t in transform_test.transforms if not isinstance(t, transforms.Normalize)])

    full_train = dataset_cls(root=DATASET_CACHE_DIR, train=True, download=True, transform=transform_train)
    full_test = dataset_cls(root=DATASET_CACHE_DIR, train=False, download=True, transform=transform_test)

    return {
        "train_loader": DataLoader(full_train, batch_size=batch_size, shuffle=True),
        "test_loader": DataLoader(full_test, batch_size=batch_size, shuffle=False),
        "dataset_name": dataset_name,
        "input_shape": input_shape,
        "input_size": int(np.prod(input_shape)),
        "num_classes": num_classes,
        "train_samples": len(full_train),
        "test_samples": len(full_test),
        "task": "classification",
    }


def _split_tensor_dataset(data, labels, batch_size, dataset_name,
                           input_shape, num_classes, task):
    """Split a TensorDataset into train/test loaders and return metadata."""
    num_samples = len(data)
    split = int(DEFAULT_TRAIN_SPLIT * num_samples)

    ds = torch.utils.data.TensorDataset(data, labels)
    train_ds = torch.utils.data.Subset(ds, range(split))
    test_ds = torch.utils.data.Subset(ds, range(split, num_samples))

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "dataset_name": dataset_name,
        "input_shape": input_shape,
        "input_size": int(np.prod(input_shape)),
        "num_classes": num_classes,
        "train_samples": split,
        "test_samples": num_samples - split,
        "task": task,
    }


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------


def _load_mnist(batch_size, params):
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return _load_torchvision_dataset(
        torchvision.datasets.MNIST, params, batch_size,
        transform_train, transform_test, "MNIST", [1, 28, 28], 10,
    )


def _load_cifar10(batch_size, params):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    transform_train = transforms.Compose(aug + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    return _load_torchvision_dataset(
        torchvision.datasets.CIFAR10, params, batch_size,
        transform_train, transform_test, "CIFAR-10", [3, 32, 32], 10,
    )


def _load_cifar100(batch_size, params):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    transform_train = transforms.Compose(aug + [transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    return _load_torchvision_dataset(
        torchvision.datasets.CIFAR100, params, batch_size,
        transform_train, transform_test, "CIFAR-100", [3, 32, 32], 100,
    )


def _generate_xor(batch_size, params):
    num_samples = params.get("num_samples", 1000)
    noise = params.get("noise_level", 0.05)

    np.random.seed(XOR_RANDOM_SEED)
    centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    labels_center = np.array([0, 1, 1, 0])

    indices = np.random.randint(0, 4, num_samples)
    data = centers[indices] + np.random.randn(num_samples, 2) * noise
    labels = labels_center[indices]

    return _split_tensor_dataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        batch_size, "XOR", [2], 2, "classification",
    )


def _load_fashion_mnist(batch_size, params):
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    return _load_torchvision_dataset(
        torchvision.datasets.FashionMNIST, params, batch_size,
        transform_train, transform_test, "Fashion-MNIST", [1, 28, 28], 10,
    )


def _generate_synthetic_regression(batch_size, params):
    num_samples = params.get("num_samples", 1000)
    noise = params.get("noise_level", 0.05)
    input_dim = params.get("input_dim", 10)
    seed = params.get("seed", 42)

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (num_samples, input_dim)).astype(np.float32)
    y = np.sin(2 * np.pi * x.sum(axis=1)) + rng.normal(0, noise, num_samples).astype(np.float32)

    return _split_tensor_dataset(
        torch.tensor(x), torch.tensor(y).unsqueeze(1),
        batch_size, "Synthetic Regression", [input_dim], 1, "regression",
    )


def _generate_polynomial(batch_size, params):
    num_samples = params.get("num_samples", 1000)
    degree = params.get("degree", 3)
    noise = params.get("noise_level", 0.1)
    input_dim = params.get("input_dim", 1)

    np.random.seed(POLYNOMIAL_RANDOM_SEED)
    x = np.random.uniform(-2, 2, (num_samples, input_dim)).astype(np.float32)

    y = np.zeros(num_samples, dtype=np.float32)
    for d in range(degree + 1):
        coeffs = np.random.randn(input_dim).astype(np.float32) * (0.5 / (d + 1))
        y += np.sum(coeffs * (x ** d), axis=1)

    y += np.random.randn(num_samples).astype(np.float32) * noise

    return _split_tensor_dataset(
        torch.tensor(x), torch.tensor(y).unsqueeze(1),
        batch_size, "Polynomial", [input_dim], 1, "regression",
    )
