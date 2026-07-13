import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, x, y, title, filename):
    x_min, x_max = x[:, 0].min() - 0.25, x[:, 0].max() + 0.25
    y_min, y_max = x[:, 1].min() - 0.25, x[:, 1].max() + 0.25
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.column_stack((xx.ravel(), yy.ravel()))
    zz = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, zz, levels=30, cmap="RdBu", alpha=0.75)
    plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap="RdBu", edgecolor="white", s=35)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
