"""
delta.py
Widrow-Hoff (Delta rule) implemented from scratch.

Rule:   y_hat = sum_i w_i * x_i + b
        w_i  += eta * (t - y_hat) * x_i
        b    += eta * (t - y_hat)

This is gradient descent on the squared error of a LINEAR neuron.
It succeeds where Hebb fails because the update is driven by the ERROR,
not by the raw co-activation of input and output.

Run:  python src/delta.py
"""

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_iris_2d, load_logic_gates


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DeltaNeuron:
    """Linear neuron trained with the Widrow-Hoff rule."""

    def __init__(self, n_inputs: int, eta: float = 0.01):
        # small random init so the first gradient is not degenerate
        rng = np.random.default_rng(0)
        self.w = rng.normal(0, 0.01, size=n_inputs)
        self.b = 0.0
        self.eta = eta

    def forward(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.b)

    def update(self, x: np.ndarray, t: float) -> float:
        """One delta step. Returns the squared error for this sample."""
        y = self.forward(x)
        err = t - y
        self.w += self.eta * err * x
        self.b += self.eta * err
        return err ** 2


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_delta(X: np.ndarray, y: np.ndarray,
                eta: float = 0.01, epochs: int = 100,
                verbose_every: int = 10):
    """
    Sequential (per-sample) Widrow-Hoff. This is what the original
    formulation specifies. Batch variants exist but per-sample is
    the canonical Delta rule.

    Returns (neuron, history) where history["mse"] is the epoch-wise MSE.
    """
    neuron = DeltaNeuron(n_inputs=X.shape[1], eta=eta)
    history = {"mse": []}

    for ep in range(1, epochs + 1):
        total_se = 0.0
        for xi, yi in zip(X, y):
            total_se += neuron.update(xi, float(yi))
        mse = total_se / len(X)
        history["mse"].append(mse)

        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            print(f"  epoch {ep:3d}  MSE = {mse:.5f}")

    return neuron, history


def evaluate(neuron: DeltaNeuron, X: np.ndarray, y: np.ndarray):
    """
    Classification decision: sign(y_hat) vs target {-1, +1}.
    Also returns raw outputs for the decision-boundary plot.
    """
    raw = np.array([neuron.forward(xi) for xi in X])
    preds = np.where(raw >= 0, 1, -1)
    acc = float(np.mean(preds == y))
    return raw, preds, acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_mse_curve(history: dict, title: str,
                   save_as: str = None,
                   out_dir: str = "results/figures"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["mse"], color="tab:blue", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(os.path.join(out_dir, save_as), dpi=150)
    plt.show()


def plot_decision_boundary_2d(neuron: DeltaNeuron,
                              X: np.ndarray, y: np.ndarray,
                              title: str, save_as: str = None,
                              out_dir: str = "results/figures"):
    """Only meaningful for 2-feature data (our Iris subset)."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, color in [(-1, "#d62728"), (1, "#2ca02c")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, edgecolors="k", s=60,
                   label=f"class {cls}")

    # boundary:  w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
    x1s = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    if abs(neuron.w[1]) > 1e-9:
        x2s = -(neuron.w[0] * x1s + neuron.b) / neuron.w[1]
        ax.plot(x1s, x2s, "b--", linewidth=1.8, label="boundary")

    ax.set_xlabel("feature 1 (standardized)")
    ax.set_ylabel("feature 2 (standardized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(os.path.join(out_dir, save_as), dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def experiment_iris():
    """Main Delta rule demo: 2-feature, 2-class Iris."""
    print("\n=== Delta rule on Iris (features 0,1; classes setosa/versicolor) ===")

    train_loader, test_loader, _ = load_iris_2d(
        feature_idx=(0, 1), class_pair=(0, 1), test_size=0.2, seed=42
    )

    # Unpack DataLoaders back into arrays (dataset is tiny)
    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    X_test  = test_loader.dataset.tensors[0].numpy()
    y_test  = test_loader.dataset.tensors[1].numpy()

    neuron, history = train_delta(X_train, y_train, eta=0.01, epochs=100,
                                  verbose_every=20)

    _, _, acc_train = evaluate(neuron, X_train, y_train)
    _, _, acc_test  = evaluate(neuron, X_test,  y_test)

    print(f"\nFinal weights: w=({neuron.w[0]:+.3f}, {neuron.w[1]:+.3f}), b={neuron.b:+.3f}")
    print(f"Train accuracy: {acc_train*100:.1f}%")
    print(f"Test  accuracy: {acc_test*100:.1f}%")

    plot_mse_curve(history, "Delta rule – Iris MSE per epoch",
                   save_as="delta_iris_mse.png")
    plot_decision_boundary_2d(
        neuron, X_train, y_train,
        "Delta rule decision boundary (train set)",
        save_as="delta_iris_boundary.png"
    )
    return neuron, history, (acc_train, acc_test)


def experiment_learning_rate_sweep():
    """Show how eta affects convergence speed."""
    print("\n=== Learning rate sweep (Iris) ===")
    train_loader, test_loader, _ = load_iris_2d()
    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()

    plt.figure(figsize=(8, 5))
    for eta in [0.001, 0.005, 0.01, 0.05, 0.1]:
        _, hist = train_delta(X_train, y_train, eta=eta, epochs=100,
                              verbose_every=0)
        plt.plot(hist["mse"], label=f"eta = {eta}", linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Delta rule convergence vs learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/delta_lr_sweep.png", dpi=150)
    plt.show()


def experiment_hebb_vs_delta_on_xor():
    """
    Side-by-side fairness demo:
    Hebb fails on XOR, Delta also fails on XOR (because XOR is not
    linearly separable). The point is that Delta can SEPARATE more
    complex boundaries than Hebb — but neither solves XOR.
    """
    print("\n=== Delta rule on XOR (a linear model cannot solve this) ===")
    X_t, y_t = load_logic_gates("XOR")
    X = X_t.numpy()
    y = y_t.numpy()

    neuron, history = train_delta(X, y, eta=0.1, epochs=200, verbose_every=50)
    _, preds, acc = evaluate(neuron, X, y)
    print(f"XOR accuracy with Delta: {acc*100:.1f}%  (expected: 50-75%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from train_utils import set_seed
    set_seed(42)

    experiment_iris()
    experiment_learning_rate_sweep()
    experiment_hebb_vs_delta_on_xor()