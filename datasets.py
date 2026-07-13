import numpy as np


def make_xor():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x, y


def make_spiral(n_points_per_class=200, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    x = []
    y = []

    for class_id in range(2):
        radius = np.linspace(0.0, 1.0, n_points_per_class)
        theta = (
            np.linspace(class_id * np.pi, (class_id + 1) * np.pi, n_points_per_class)
            + rng.normal(0.0, noise, n_points_per_class)
        )
        points = np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))
        x.append(points)
        y.append(np.full(n_points_per_class, class_id))

    x = np.vstack(x)
    y = np.concatenate(y).reshape(-1, 1)
    indices = rng.permutation(len(x))
    return x[indices], y[indices].astype(float)
