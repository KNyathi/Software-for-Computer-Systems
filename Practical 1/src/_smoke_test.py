from train_utils import set_seed, get_device
from data_utils import (load_logic_gates, load_iris_2d,
                        load_mnist, make_sine_windows)

set_seed(42)
print("device:", get_device())

X, y = load_logic_gates("AND")
print("AND X:", X.tolist(), "y:", y.tolist())

tr, te, scaler = load_iris_2d()
xb, yb = next(iter(tr))
print("Iris batch:", xb.shape, yb.shape)

tr, te = load_mnist(batch_size=64)
xb, yb = next(iter(tr))
print("MNIST batch:", xb.shape, yb.shape)

tr, te, (series, ns) = make_sine_windows()
xb, yb = next(iter(tr))
print("Sine batch:", xb.shape, yb.shape)