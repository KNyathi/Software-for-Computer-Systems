"""
train_utils.py
Shared utilities: reproducibility, device selection, logging, plotting.
Every other module imports from here so results are comparable across runs.
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Fix every source of randomness we touch.
    Call this ONCE at the top of every script / notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN – slightly slower but reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Pick GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():       # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------
def save_history(history: dict, name: str, out_dir: str = "results/logs") -> str:
    """
    Persist a training history dict as JSON.
    history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return path


def load_history(name: str, in_dir: str = "results/logs") -> dict:
    with open(os.path.join(in_dir, f"{name}.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_curves(history: dict, title: str, ylabel: str,
                save_as: str = None, out_dir: str = "results/figures"):
    """
    Plot one or more metric curves on a single axis.
    `history` maps label -> list of values, e.g. {"train": [...], "val": [...]}
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in history.items():
        plt.plot(values, label=label, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(os.path.join(out_dir, save_as), dpi=150)
    plt.show()