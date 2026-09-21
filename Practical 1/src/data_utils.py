"""
data_utils.py
Dataset loaders for all four algorithms.

  - load_logic_gates()  -> Hebb / Delta toy data (bipolar AND, OR, XOR)
  - load_iris_2d()      -> Delta rule on 2 features, 2 classes
  - load_mnist()        -> CNN
  - make_sine_windows() -> RNN
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# 1. Logic gates (Hebb + Delta sanity check)
# ---------------------------------------------------------------------------
def load_logic_gates(gate: str = "AND") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (X, y) with bipolar inputs {-1, +1} and targets {-1, +1}.
    This is the representation Hebb's rule requires.

    AND:  y = +1 only when both inputs are +1
    OR:   y = +1 when at least one input is +1
    XOR:  y = +1 when inputs differ  (Hebb CANNOT learn this)
    """
    X = np.array([
        [-1, -1],
        [-1, +1],
        [+1, -1],
        [+1, +1],
    ], dtype=np.float32)

    gate = gate.upper()
    if gate == "AND":
        y = np.array([-1, -1, -1, +1], dtype=np.float32)
    elif gate == "OR":
        y = np.array([-1, +1, +1, +1], dtype=np.float32)
    elif gate == "XOR":
        y = np.array([-1, +1, +1, -1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown gate: {gate}")

    return torch.from_numpy(X), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# 2. Iris, reduced to 2 features / 2 classes (Delta rule)
# ---------------------------------------------------------------------------
def load_iris_2d(feature_idx=(0, 1), class_pair=(0, 1),
                 test_size: float = 0.2, seed: int = 42):
    """
    Load Iris, keep two features and two classes, standardize, split.
    Returns DataLoaders ready for a linear neuron.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # keep only the two requested classes
    mask = np.isin(y, class_pair)
    X, y = X[mask], y[mask]

    # keep only the two requested features
    X = X[:, list(feature_idx)]

    # remap labels to {-1, +1} for the Widrow-Hoff rule
    y = np.where(y == class_pair[0], -1.0, +1.0).astype(np.float32)

    # standardize (z-score) — critical for stable learning-rate behavior
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)
    return train_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# 3. MNIST (CNN)
# ---------------------------------------------------------------------------
def load_mnist(batch_size: int = 64, root: str = "data/raw"):
    """
    Standard MNIST with the canonical normalization.
    Returns (train_loader, test_loader).
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Sine wave -> sliding windows (RNN)
# ---------------------------------------------------------------------------
def make_sine_windows(n_points: int = 5000,
                      seq_len: int = 30,
                      noise_std: float = 0.1,
                      test_size: float = 0.2,
                      seed: int = 42,
                      batch_size: int = 32):
    """
    Generate a noisy sine wave, cut it into (seq_len -> next value) windows.

    Returns:
        train_loader, test_loader, (series, noise_std)
    Each batch has shape:
        X: (B, seq_len, 1)   y: (B, 1)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 100, n_points)
    series = np.sin(t) + noise_std * rng.standard_normal(n_points)
    series = series.astype(np.float32)

    # Build sliding windows
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # Chronological split — never shuffle time series!
    n_train = int(len(X) * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, (series, noise_std)