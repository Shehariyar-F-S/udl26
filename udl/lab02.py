"""Helpers used only in Lab 2 (week02): data and plots. Provided - you do not implement these."""
import pickle
from pathlib import Path

import numpy as np
import torch

from udl.plotting import BLUE, ORANGE, to_numpy

DATA_FILE = Path(__file__).parent / "data" / "lab02_data.pkl"


# ---- data ----------------------------------------------------------------------------------

def _load_data():
    """The pickle holds a dict of numpy arrays (nothing else)."""
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


def load_categorical_1d():
    """One discrete variable with d = 20 values and two bumps. Returns train (2000,), test (5000,) uint8 tensors and d."""
    d = 20
    k = torch.arange(d)

    def bump(m, s):
        b = torch.exp(-0.5 * ((k - m) / s) ** 2)
        return b / b.sum()

    probs = 0.55 * bump(5, 1.6) + 0.45 * bump(14, 2.2) + 0.0015
    g = torch.Generator().manual_seed(0)
    x = torch.multinomial(probs, 7000, replacement=True, generator=g).to(torch.uint8)
    return x[:2000], x[2000:], d


def sample_gaussian_2d(n, seed=0, rho=0.7, limit=3.0):
    """n points of a standard 2D Gaussian with correlation rho, truncated to the square [-limit, limit]^2. Float64 tensor (n, 2)."""
    g = torch.Generator().manual_seed(seed)
    L = torch.linalg.cholesky(torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64))
    x = torch.randn(2 * n, 2, generator=g, dtype=torch.float64) @ L.T
    return x[(x.abs() < limit).all(dim=1)][:n]


def load_2d(which):
    """Discrete 2D data: pairs of integers in 0..d-1.  which = 1 (d = 25) or 2 (d = 200).
    Returns train and test as uint8 tensors (N, 2) and d. Dataset 1 has 50000 + 12500 examples, dataset 2 has 200000 + 50000."""
    z = _load_data()
    return torch.from_numpy(z[f"d{which}_train"]), torch.from_numpy(z[f"d{which}_test"]), int(z[f"d{which}_d"])


def load_shapes():
    """Binary 20 x 20 images of simple shapes (1 = ink). Returns train (10479, 20, 20), test (4491, 20, 20) as uint8 tensors."""
    z = _load_data()
    return torch.from_numpy(z["shapes_train"]), torch.from_numpy(z["shapes_test"])


def load_binary_mnist():
    """Binary 28 x 28 MNIST (pixel is 1 if brighter than half, as in Lab 1). Returns train (60000, 28, 28), test (10000, 28, 28) as uint8 tensors."""
    from udl.datasets import load_mnist

    train, test = load_mnist()
    return (train > 0.5).to(torch.uint8), (test > 0.5).to(torch.uint8)


def count_table(x, d):
    """Histogram of 2D integer data as a (d, d) array: table[x0, x1] = share of examples."""
    t = np.zeros((d, d))
    np.add.at(t, (to_numpy(x[:, 0]).astype(int), to_numpy(x[:, 1]).astype(int)), 1.0)
    return t / t.sum()


# ---- evaluation and plots ------------------------------------------------------------------

def report(name, test_losses, unit):
    """Print the last test loss."""
    print(f"{name}: test NLL = {float(test_losses[-1]):.4f} {unit}")


def plot_curves(ax, train_losses, test_losses, ylabel):
    """Training loss per minibatch (blue) and test loss at initialisation and after every epoch (orange)."""
    train_losses, test_losses = to_numpy(train_losses), to_numpy(test_losses)
    n_epochs = len(test_losses) - 1
    per_epoch = len(train_losses) / n_epochs
    ax.plot(np.arange(1, len(train_losses) + 1) / per_epoch, train_losses, color=BLUE, lw=1, label="train (minibatch)")
    ax.plot(np.arange(n_epochs + 1), test_losses, color=ORANGE, lw=2, marker="o", ms=3, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_table(ax, table, title=""):
    """Heatmap of a (d, d) table indexed [x0, x1]: x0 horizontal, x1 vertical."""
    ax.imshow(table.T, origin="lower", cmap="viridis")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)


def plot_categorical_fit(ax, train, d, probs=None, title=""):
    """Relative frequencies of the training data (blue bars, counting) and, if given, the learned probabilities (orange markers)."""
    freq = np.bincount(to_numpy(train).astype(int), minlength=d) / len(train)
    ax.bar(np.arange(d), freq, color=BLUE, edgecolor="#163c69", alpha=0.8, label="data (counting)")
    if probs is not None:
        ax.plot(np.arange(d), to_numpy(probs), "o", color=ORANGE, ms=5, label="learned")
    ax.set_xticks(np.arange(0, d, 2))
    ax.set_xlabel("x")
    ax.set_ylabel("probability")
    ax.set_title(title)
    if probs is not None:
        ax.legend(fontsize=8)


def plot_2d_result(axes, train, probs, d, name=""):
    """Two heatmaps on axes[0:2]: the training data and the learned (d, d) probabilities."""
    plot_table(axes[0], count_table(train, d), f"{name}: training data")
    plot_table(axes[1], to_numpy(probs), f"{name}: learned p(x0, x1)")


def show_images(ax, images, nrows, ncols, title=""):
    """Grid of nrows x ncols binary images (N, H, W), white ink on black."""
    im = to_numpy(images[: nrows * ncols]).astype(float)
    n, h, w = im.shape
    im = np.pad(im, ((0, nrows * ncols - n), (1, 1), (1, 1)), constant_values=0.5)
    h, w = h + 2, w + 2
    grid = im.reshape(nrows, ncols, h, w).transpose(0, 2, 1, 3).reshape(nrows * h, ncols * w)
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
