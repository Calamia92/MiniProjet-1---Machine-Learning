import argparse

import numpy as np

from datasets import make_spiral, make_xor
from mlp_numpy import MLP
from plotting import plot_decision_boundary, plot_loss_curve


def run_xor():
    x, y = make_xor()
    model = MLP(
        layer_sizes=[2, 4, 1],
        activations=["tanh", "sigmoid"],
        learning_rate=0.5,
        seed=7,
    )

    history = model.fit(x, y, epochs=10_000, verbose_every=1_000)
    proba = model.predict_proba(x)
    predictions = model.predict(x)

    print("\nResultats XOR")
    for inputs, target, p, pred in zip(x, y.ravel(), proba.ravel(), predictions.ravel()):
        print(f"{inputs.astype(int).tolist()} -> cible={int(target)} proba={p:.4f} pred={int(pred)}")
    print(f"accuracy={np.mean(predictions == y):.3f}")

    filename = "xor_loss.png"
    plot_loss_curve(history, "PMC NumPy - Loss XOR", filename)
    print(f"Courbe de loss sauvegardee: {filename}")


def run_spiral():
    x, y = make_spiral(n_points_per_class=250, noise=0.22, seed=12)
    model = MLP(
        layer_sizes=[2, 32, 32, 1],
        activations=["tanh", "tanh", "sigmoid"],
        learning_rate=0.08,
        seed=21,
    )

    history = model.fit(x, y, epochs=4_000, batch_size=64, verbose_every=500)
    predictions = model.predict(x)
    accuracy = np.mean(predictions == y)
    print(f"\nSpirale 2D accuracy={accuracy:.3f}")

    filename = "spiral_decision_boundary.png"
    plot_decision_boundary(model, x, y, "PMC NumPy - Spirale 2D", filename)
    print(f"Frontiere de decision sauvegardee: {filename}")

    filename = "spiral_loss.png"
    plot_loss_curve(history, "PMC NumPy - Loss Spirale 2D", filename)
    print(f"Courbe de loss sauvegardee: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Mini-projet PMC NumPy et Keras MNIST")
    parser.add_argument(
        "demo",
        choices=["xor", "spiral", "all"],
        nargs="?",
        default="all",
        help="Experience a lancer",
    )
    args = parser.parse_args()

    if args.demo in {"xor", "all"}:
        run_xor()
    if args.demo in {"spiral", "all"}:
        run_spiral()


if __name__ == "__main__":
    main()
