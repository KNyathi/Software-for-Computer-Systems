"""
cnn.py
Convolutional neural network for MNIST, built in PyTorch.

Four experiments:
  A. baseline           - 2 conv blocks + dropout
  B. no_dropout         - shows overfitting
  C. one_conv           - shows underfitting from reduced capacity
  D. sgd_momentum       - shows optimizer sensitivity

Outputs (to results/):
  logs/cnn_<name>.json           - per-epoch train/val loss & val acc
  figures/cnn_<name>_curves.png  - loss curves
  figures/cnn_baseline_cm.png    - confusion matrix
  figures/cnn_baseline_errors.png- grid of misclassified digits
  figures/cnn_ablation_summary.png - final accuracies bar chart

Run:  python src/cnn.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from train_utils import set_seed, get_device, save_history
from data_utils import load_mnist


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    Configurable CNN so we can run ablations from one class.

    conv1_channels:  out-channels of first conv block
    conv2_channels:  0 means "skip the second block"
    dropout_p:       dropout probability before the final Linear
    """

    def __init__(self,
                 conv1_channels: int = 32,
                 conv2_channels: int = 64,
                 fc_hidden: int = 128,
                 dropout_p: float = 0.5,
                 n_classes: int = 10):
        super().__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels

        # Block 1
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(conv1_channels)

        # Block 2 (optional)
        self.conv2 = None
        self.bn2   = None
        if conv2_channels > 0:
            self.conv2 = nn.Conv2d(conv1_channels, conv2_channels,
                                   kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(conv2_channels)

        # Determine flattened size after conv stack
        #   input 28x28
        #   after pool1: 14x14
        #   after pool2 (if present): 7x7
        last_ch = conv2_channels if conv2_channels > 0 else conv1_channels
        spatial = 7 if conv2_channels > 0 else 14
        flat_dim = last_ch * spatial * spatial

        self.dropout = nn.Dropout(dropout_p)
        self.fc1     = nn.Linear(flat_dim, fc_hidden)
        self.fc2     = nn.Linear(fc_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        if self.conv2 is not None:
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Train / evaluate one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += xb.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += xb.size(0)

        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return running_loss / total, correct / total, all_preds, all_targets


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_experiment(name: str,
                   model_kwargs: dict,
                   optimizer_name: str = "adam",
                   lr: float = 1e-3,
                   epochs: int = 10,
                   batch_size: int = 64,
                   device: torch.device = None):
    """
    Train one CNN configuration and persist history.
    Returns (model, history, (test_loss, test_acc, preds, targets)).
    """
    if device is None:
        device = get_device()
    print(f"\n=== Experiment: {name} ===")
    print(f"  device={device}  model={model_kwargs}  "
          f"optim={optimizer_name} lr={lr} epochs={epochs}")

    train_loader, test_loader = load_mnist(batch_size=batch_size)
    model = SimpleCNN(**model_kwargs).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(optimizer_name)

    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion, device)
        va_loss, va_acc, _, _ = evaluate(model, test_loader,
                                         criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  epoch {ep:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

    elapsed = time.time() - t0
    print(f"  -> finished in {elapsed:.1f}s")

    save_history(history, f"cnn_{name}")

    # Final test evaluation with predictions (for confusion matrix)
    test_loss, test_acc, preds, targets = evaluate(
        model, test_loader, criterion, device)
    print(f"  -> final test accuracy: {test_acc*100:.2f}%")

    return model, history, (test_loss, test_acc, preds, targets)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train", linewidth=1.8)
    axes[0].plot(history["val_loss"],   label="val",   linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"Loss — {name}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_acc"], color="tab:green", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title(f"Val accuracy — {name}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/figures/cnn_{name}_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(preds, targets, save_as="cnn_baseline_cm.png"):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — baseline CNN")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"results/figures/{save_as}", dpi=150)
    plt.show()
    return cm


def plot_misclassified(model, test_loader, device, n=10,
                       save_as="cnn_baseline_errors.png"):
    """Grid of the first n misclassified test images."""
    model.eval()
    wrong_images, wrong_true, wrong_pred = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            mask = preds != yb
            if mask.any():
                wrong_images.append(xb[mask].cpu())
                wrong_true.append(yb[mask].cpu())
                wrong_pred.append(preds[mask].cpu())
            if sum(len(w) for w in wrong_images) >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_true   = torch.cat(wrong_true)[:n]
    wrong_pred   = torch.cat(wrong_pred)[:n]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, img, t, p in zip(axes.flat, wrong_images, wrong_true, wrong_pred):
        # undo MNIST normalization for display
        ax.imshow(img.squeeze().numpy() * 0.3081 + 0.1307, cmap="gray")
        ax.set_title(f"true={t.item()}  pred={p.item()}", fontsize=10)
        ax.axis("off")
    plt.suptitle("Misclassified examples — baseline CNN")
    plt.tight_layout()
    plt.savefig(f"results/figures/{save_as}", dpi=150)
    plt.show()


def plot_ablation_summary(results: dict):
    """Bar chart of final test accuracy per configuration."""
    names  = list(results.keys())
    accs   = [results[n][1] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, accs, color="tab:blue", edgecolor="k")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("CNN ablation summary")
    ax.set_ylim(min(accs) - 1, 100)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.1,
                f"{a:.2f}%", ha="center", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/figures/cnn_ablation_summary.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # ---- Configs -----------------------------------------------------------
    EXPERIMENTS = {
        "baseline": dict(
            model_kwargs=dict(conv1_channels=32, conv2_channels=64,
                              fc_hidden=128, dropout_p=0.5),
            optimizer_name="adam", lr=1e-3, epochs=10, batch_size=64,
        ),
        "no_dropout": dict(
            model_kwargs=dict(conv1_channels=32, conv2_channels=64,
                              fc_hidden=128, dropout_p=0.0),
            optimizer_name="adam", lr=1e-3, epochs=10, batch_size=64,
        ),
        "one_conv": dict(
            model_kwargs=dict(conv1_channels=32, conv2_channels=0,
                              fc_hidden=128, dropout_p=0.5),
            optimizer_name="adam", lr=1e-3, epochs=10, batch_size=64,
        ),
        "sgd_momentum": dict(
            model_kwargs=dict(conv1_channels=32, conv2_channels=64,
                              fc_hidden=128, dropout_p=0.5),
            optimizer_name="sgd", lr=1e-2, epochs=10, batch_size=64,
        ),
    }
    # ------------------------------------------------------------------------

    results = {}
    models  = {}
    for name, cfg in EXPERIMENTS.items():
        model, history, final = run_experiment(name=name, device=device, **cfg)
        results[name] = final
        models[name]  = model
        plot_training_curves(history, name)

    # Summary
    print("\n=== Summary ===")
    for name, (loss, acc, _, _) in results.items():
        print(f"  {name:15s}  test_acc={acc*100:.2f}%  test_loss={loss:.4f}")

    plot_ablation_summary(results)

    # Extra figures only for the baseline
    baseline_model = models["baseline"]
    _, test_loader = load_mnist(batch_size=64)
    _, _, preds, targets = results["baseline"]
    plot_confusion_matrix(preds, targets)
    plot_misclassified(baseline_model, test_loader, device, n=10)

    # Save the best model so later sessions don't have to retrain
    os.makedirs("results", exist_ok=True)
    torch.save(baseline_model.state_dict(), "results/cnn_baseline.pt")
    print("\nSaved baseline weights to results/cnn_baseline.pt")