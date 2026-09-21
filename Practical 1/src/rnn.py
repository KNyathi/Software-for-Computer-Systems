"""
rnn.py
Recurrent networks for 1-step-ahead sine-wave prediction, in PyTorch.

Three cell types compared under identical conditions:
    vanilla RNN  - no gating; suffers from vanishing gradients
    LSTM         - input/forget/output gates + cell state
    GRU          - simplified two-gate variant of LSTM

Experiments:
  1. cell comparison        RNN vs LSTM vs GRU (seq_len=30, noise=0.1)
  2. sequence-length sweep  [10, 30, 60, 90] on the best two cells
  3. noise sweep            [0.0, 0.1, 0.3] on the best cell
  4. hidden-size sweep      [8, 16, 32, 64]

Outputs:
  logs/rnn_<name>.json              per-epoch train/val MSE
  figures/rnn_cell_comparison.png   three MSE curves
  figures/rnn_prediction.png        predicted vs true on test window
  figures/rnn_seqlen_sweep.png      MSE vs sequence length
  figures/rnn_noise_sweep.png       MSE vs noise level
  figures/rnn_hidden_sweep.png      MSE vs hidden size
  results/rnn_best.pt               best model state dict

Run:  python src/rnn.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

from train_utils import set_seed, get_device, save_history
from data_utils import make_sine_windows


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SineRNN(nn.Module):
    """
    One recurrent layer + a linear head producing a single scalar.

    rnn_type: "RNN" | "LSTM" | "GRU"
    """

    def __init__(self,
                 rnn_type: str = "RNN",
                 input_size: int = 1,
                 hidden_size: int = 32,
                 num_layers: int = 1,
                 output_size: int = 1):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, 1)
        # We only need the last time-step's output for 1-step prediction.
        # out: (B, T, H);  h_n shape depends on cell type
        out, _ = self.rnn(x)
        last = out[:, -1, :]        # (B, H)
        return self.head(last)      # (B, 1)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_se, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total_se += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_se / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_se, n = 0.0, 0
    preds_all, targets_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)

        total_se += loss.item() * xb.size(0)
        n += xb.size(0)
        preds_all.append(pred.cpu())
        targets_all.append(yb.cpu())
    preds_all   = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()
    return total_se / n, preds_all, targets_all


# ---------------------------------------------------------------------------
# One full experiment
# ---------------------------------------------------------------------------
def run_experiment(name: str,
                   rnn_type: str = "RNN",
                   hidden_size: int = 32,
                   num_layers: int = 1,
                   seq_len: int = 30,
                   noise_std: float = 0.1,
                   lr: float = 1e-3,
                   epochs: int = 50,
                   batch_size: int = 32,
                   device: torch.device = None,
                   verbose: bool = True):
    """
    Train one configuration. Returns (model, history, (test_mse, preds, targets)).
    """
    if device is None:
        device = get_device()

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  cell={rnn_type}  hidden={hidden_size}  layers={num_layers}  "
              f"seq_len={seq_len}  noise={noise_std}  lr={lr}  epochs={epochs}")

    train_loader, test_loader, _ = make_sine_windows(
        n_points=5000, seq_len=seq_len, noise_std=noise_std,
        batch_size=batch_size, seed=42
    )

    model = SineRNN(rnn_type=rnn_type, hidden_size=hidden_size,
                    num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_mse": [], "val_mse": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va, _, _ = evaluate(model, test_loader, criterion, device)
        history["train_mse"].append(tr)
        history["val_mse"].append(va)

        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            print(f"  epoch {ep:3d}/{epochs}  train_mse={tr:.6f}  val_mse={va:.6f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"  -> finished in {elapsed:.1f}s  final val_mse={history['val_mse'][-1]:.6f}")

    # Save log
    save_history(history, f"rnn_{name}")

    # Final evaluation with predictions
    test_mse, preds, targets = evaluate(model, test_loader, criterion, device)
    return model, history, (test_mse, preds, targets)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(curves: dict, title: str, save_as: str,
                ylabel: str = "MSE (log scale)", logy: bool = True):
    """curves: {label: list_of_values}"""
    os.makedirs("results/figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        plt.plot(values, label=label, linewidth=1.8)
    if logy:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"results/figures/{save_as}", dpi=150)
    plt.show()


def plot_prediction(preds: np.ndarray, targets: np.ndarray,
                    zoom: int = 200, save_as: str = "rnn_prediction.png"):
    """Predicted vs true sine on the test set (zoomed to first `zoom` points)."""
    plt.figure(figsize=(11, 4.5))
    plt.plot(targets[:zoom], label="true", linewidth=1.8)
    plt.plot(preds[:zoom],   label="predicted", linewidth=1.8, linestyle="--")
    plt.xlabel("Test time step")
    plt.ylabel("Value")
    plt.title(f"Prediction vs truth (first {zoom} test points)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/{save_as}", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def experiment_cell_comparison(device):
    """RNN vs LSTM vs GRU under identical conditions."""
    print("\n########## Experiment 1: cell comparison ##########")
    configs = {
        "RNN":  dict(rnn_type="RNN"),
        "LSTM": dict(rnn_type="LSTM"),
        "GRU":  dict(rnn_type="GRU"),
    }
    curves = {}
    summary = {}
    best_model, best_mse, best_name = None, float("inf"), None

    for name, cfg in configs.items():
        model, hist, (mse, preds, targets) = run_experiment(
            name=f"cell_{name}", hidden_size=32, seq_len=30, noise_std=0.1,
            epochs=50, device=device, **cfg)
        curves[name] = hist["val_mse"]
        summary[name] = mse

        if mse < best_mse:
            best_mse, best_model, best_name = mse, model, name
            best_preds, best_targets = preds, targets

    plot_curves(curves, "RNN vs LSTM vs GRU — validation MSE",
                "rnn_cell_comparison.png")
    plot_prediction(best_preds, best_targets, zoom=200,
                    save_as="rnn_prediction.png")

    print("\n--- Cell comparison summary (test MSE) ---")
    for k, v in summary.items():
        print(f"  {k:5s}  {v:.6f}")

    torch.save(best_model.state_dict(), "results/rnn_best.pt")
    print(f"Saved best model ({best_name}) to results/rnn_best.pt")
    return summary, best_name


def experiment_seqlen_sweep(device):
    """Does longer context break vanilla RNN? It should."""
    print("\n########## Experiment 2: sequence-length sweep ##########")
    lengths = [10, 30, 60, 90]
    summary = {"RNN": [], "LSTM": []}

    for L in lengths:
        for cell in ["RNN", "LSTM"]:
            _, _, (mse, _, _) = run_experiment(
                name=f"seqlen_{cell}_{L}",
                rnn_type=cell, hidden_size=32,
                seq_len=L, noise_std=0.1, epochs=50,
                device=device, verbose=False)
            summary[cell].append(mse)
            print(f"  seq_len={L:3d}  {cell:5s}  test_mse={mse:.6f}")

    plt.figure(figsize=(7, 4.5))
    for cell, values in summary.items():
        plt.plot(lengths, values, marker="o", linewidth=1.8, label=cell)
    plt.xlabel("Sequence length")
    plt.ylabel("Test MSE")
    plt.yscale("log")
    plt.title("Effect of sequence length on RNN vs LSTM")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/figures/rnn_seqlen_sweep.png", dpi=150)
    plt.show()
    return summary


def experiment_noise_sweep(device):
    """How robust is the model to input noise?"""
    print("\n########## Experiment 3: noise sweep ##########")
    noise_levels = [0.0, 0.1, 0.3]
    summary = []

    for ns in noise_levels:
        _, _, (mse, _, _) = run_experiment(
            name=f"noise_{ns}",
            rnn_type="LSTM", hidden_size=32,
            seq_len=30, noise_std=ns, epochs=50,
            device=device, verbose=False)
        summary.append(mse)
        print(f"  noise={ns:.1f}  test_mse={mse:.6f}")

    plt.figure(figsize=(7, 4.5))
    plt.bar([str(n) for n in noise_levels], summary,
            color="tab:orange", edgecolor="k")
    plt.xlabel("Noise std")
    plt.ylabel("Test MSE")
    plt.title("LSTM robustness to input noise")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/figures/rnn_noise_sweep.png", dpi=150)
    plt.show()
    return summary


def experiment_hidden_sweep(device):
    """Where does capacity saturate?"""
    print("\n########## Experiment 4: hidden-size sweep ##########")
    sizes = [8, 16, 32, 64]
    summary = []

    for h in sizes:
        _, _, (mse, _, _) = run_experiment(
            name=f"hidden_{h}",
            rnn_type="LSTM", hidden_size=h,
            seq_len=30, noise_std=0.1, epochs=50,
            device=device, verbose=False)
        summary.append(mse)
        print(f"  hidden={h:3d}  test_mse={mse:.6f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, summary, marker="o", linewidth=1.8, color="tab:green")
    plt.xlabel("Hidden size")
    plt.ylabel("Test MSE")
    plt.yscale("log")
    plt.title("LSTM capacity vs performance")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/figures/rnn_hidden_sweep.png", dpi=150)
    plt.show()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # If you want to save time, comment out any of these.
    experiment_cell_comparison(device)
    experiment_seqlen_sweep(device)
    experiment_noise_sweep(device)
    experiment_hidden_sweep(device)

    print("\nAll RNN experiments complete.")