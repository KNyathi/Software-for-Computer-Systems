"""
hebb.py
Hebbian learning implemented from scratch (no PyTorch autograd, no nn.Module).

Rule:      w_i <- w_i + eta * x_i * y
Output:    y_hat = sign( sum_i w_i * x_i - theta )

We use bipolar inputs {-1, +1} because with {0, 1} a zero input
never contributes to the weight update, which cripples the rule.

Run:  python src/hebb.py
"""

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_logic_gates


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HebbianNeuron:
    """
    Single Hebbian neuron with a fixed threshold theta.
    Weights are updated once per pattern (one epoch is typical).
    """

    def __init__(self, n_inputs: int, eta: float = 0.1, theta: float = 0.0):
        self.w = np.zeros(n_inputs, dtype=np.float64)
        self.eta = eta
        self.theta = theta

    def predict(self, x: np.ndarray) -> int:
        """Return +1 or -1."""
        s = float(np.dot(self.w, x)) - self.theta
        return 1 if s >= 0 else -1

    def hebbian_step(self, x: np.ndarray, y: int) -> None:
        """One Hebbian update for a single (x, y) pair."""
        self.w += self.eta * x * y


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_hebb(X: np.ndarray, y: np.ndarray,
               eta: float = 0.1, theta: float = 0.0,
               epochs: int = 1):
    """
    Hebb is a one-shot rule; we still expose `epochs` to show that
    repeating the pass amplifies weights without adding information.
    """
    neuron = HebbianNeuron(n_inputs=X.shape[1], eta=eta, theta=theta)
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            neuron.hebbian_step(xi, int(yi))
    return neuron


def evaluate(neuron: HebbianNeuron, X: np.ndarray, y: np.ndarray):
    preds = np.array([neuron.predict(xi) for xi in X])
    acc = float(np.mean(preds == y))
    return preds, acc


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_gate(gate: str, eta: float = 0.1, theta: float = 0.0,
             epochs: int = 1, verbose: bool = True):
    X_t, y_t = load_logic_gates(gate)
    X = X_t.numpy()
    y = y_t.numpy().astype(int)

    neuron = train_hebb(X, y, eta=eta, theta=theta, epochs=epochs)
    preds, acc = evaluate(neuron, X, y)

    if verbose:
        print(f"\n=== Hebb on {gate}  (eta={eta}, theta={theta}, epochs={epochs}) ===")
        print(f"Final weights: w1={neuron.w[0]:+.3f}  w2={neuron.w[1]:+.3f}")
        print(f"{'x1':>4} {'x2':>4} {'target':>7} {'pred':>6} {'ok':>4}")
        for xi, yi, pi in zip(X, y, preds):
            print(f"{xi[0]:>4.0f} {xi[1]:>4.0f} {yi:>7} {pi:>6} "
                  f"{'yes' if yi == pi else 'NO':>4}")
        print(f"Accuracy: {acc*100:.1f}%")

    return neuron, preds, acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_decision_boundary(neuron: HebbianNeuron, X: np.ndarray, y: np.ndarray,
                           gate: str, save_as: str = None,
                           out_dir: str = "results/figures"):
    """Visualise which side of the hyperplane each pattern falls on."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    # scatter the four patterns
    for xi, yi in zip(X, y):
        color = "#2ca02c" if yi == 1 else "#d62728"
        ax.scatter(xi[0], xi[1], s=220, c=color,
                   edgecolors="k", zorder=3)
        ax.annotate(f"({xi[0]:+.0f},{xi[1]:+.0f})", (xi[0], xi[1]),
                    textcoords="offset points", xytext=(10, 8), fontsize=9)

    # decision boundary: w1*x1 + w2*x2 = theta  ->  x2 = (theta - w1*x1)/w2
    xs = np.linspace(-1.6, 1.6, 100)
    if abs(neuron.w[1]) > 1e-9:
        ys = (neuron.theta - neuron.w[0] * xs) / neuron.w[1]
        ax.plot(xs, ys, "b--", label="decision boundary")
    else:
        # vertical line
        if abs(neuron.w[0]) > 1e-9:
            xv = neuron.theta / neuron.w[0]
            ax.axvline(xv, color="b", linestyle="--",
                       label="decision boundary")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Hebb on {gate}   w=({neuron.w[0]:+.2f}, {neuron.w[1]:+.2f}), theta={neuron.theta}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(os.path.join(out_dir, save_as), dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from train_utils import set_seed
    set_seed(42)

    # AND and OR are linearly separable — Hebb should nail these.
    # XOR is not — Hebb must fail. That failure motivates the Delta rule.
    results = {}
    for gate, theta in [("AND", 0.5), ("OR", 0.5), ("XOR", 0.5)]:
        neuron, preds, acc = run_gate(gate, eta=0.5, theta=theta, epochs=1)
        results[gate] = acc

        X_t, y_t = load_logic_gates(gate)
        plot_decision_boundary(neuron, X_t.numpy(),
                               y_t.numpy().astype(int),
                               gate, save_as=f"hebb_{gate.lower()}.png")

    print("\n=== Hebb summary ===")
    for g, a in results.items():
        print(f"  {g}: {a*100:.1f}%")