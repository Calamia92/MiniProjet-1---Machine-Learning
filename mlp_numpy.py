import numpy as np


class MLP:
    """Perceptron multicouche from scratch avec NumPy."""

    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.1,
        seed=42,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes doit contenir au moins entree et sortie.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Il faut une activation par couche de poids.")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / n_in)
            self.weights.append(self.rng.normal(0.0, limit, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    def _activation(self, z, name):
        if name == "sigmoid":
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
        if name == "tanh":
            return np.tanh(z)
        if name == "relu":
            return np.maximum(0.0, z)
        if name == "softmax":
            shifted = z - np.max(z, axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / np.sum(exp, axis=1, keepdims=True)
        if name == "linear":
            return z
        raise ValueError(f"Activation inconnue: {name}")

    def _activation_derivative(self, activated, name):
        if name == "sigmoid":
            return activated * (1.0 - activated)
        if name == "tanh":
            return 1.0 - activated**2
        if name == "relu":
            return (activated > 0.0).astype(float)
        if name == "linear":
            return np.ones_like(activated)
        raise ValueError(f"Derivee non supportee pour: {name}")

    def forward(self, x):
        activations = [x]
        pre_activations = []

        a = x
        for w, b, activation_name in zip(self.weights, self.biases, self.activations):
            z = a @ w + b
            a = self._activation(z, activation_name)
            pre_activations.append(z)
            activations.append(a)

        return activations, pre_activations

    def predict_proba(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def predict(self, x):
        proba = self.predict_proba(x)
        if proba.shape[1] == 1:
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def fit(self, x, y, epochs=2000, batch_size=None, verbose_every=0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = x.shape[0]
        batch_size = batch_size or n_samples
        history = []

        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                stop = start + batch_size
                xb = x_shuffled[start:stop]
                yb = y_shuffled[start:stop]
                self._train_batch(xb, yb)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0):
                loss = self.loss(x, y)
                history.append((epoch, loss))
                print(f"epoch={epoch:5d} loss={loss:.6f}")

        return history

    def _train_batch(self, x, y):
        activations, _ = self.forward(x)
        m = x.shape[0]

        delta = self._output_delta(activations[-1], y)
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        for layer in reversed(range(len(self.weights))):
            grad_weights[layer] = activations[layer].T @ delta / m
            grad_biases[layer] = np.mean(delta, axis=0, keepdims=True)

            if layer > 0:
                previous_activation = activations[layer]
                delta = (
                    delta
                    @ self.weights[layer].T
                    * self._activation_derivative(previous_activation, self.activations[layer - 1])
                )

        for layer in range(len(self.weights)):
            self.weights[layer] -= self.learning_rate * grad_weights[layer]
            self.biases[layer] -= self.learning_rate * grad_biases[layer]

    def _output_delta(self, y_pred, y_true):
        output_activation = self.activations[-1]
        if output_activation in {"sigmoid", "softmax"}:
            return y_pred - y_true
        return (y_pred - y_true) * self._activation_derivative(y_pred, output_activation)

    def loss(self, x, y):
        y_pred = self.predict_proba(x)
        eps = 1e-12

        if self.activations[-1] == "sigmoid":
            y_pred = np.clip(y_pred, eps, 1.0 - eps)
            return float(-np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred)))

        if self.activations[-1] == "softmax":
            y_pred = np.clip(y_pred, eps, 1.0)
            return float(-np.mean(np.sum(y * np.log(y_pred), axis=1)))

        return float(np.mean((y_pred - y) ** 2))


def one_hot(labels, n_classes):
    labels = np.asarray(labels, dtype=int)
    encoded = np.zeros((labels.size, n_classes))
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded
