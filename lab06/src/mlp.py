"""A small NumPy MLP with explicit forward and backward passes."""
from __future__ import annotations

import numpy as np


class MLPRegressor:
    """Fully-connected tanh MLP with a linear two-output layer.

    Parameters are updated with Adam or full-batch gradient descent.  The
    implementation intentionally exposes weights and training history so the
    backpropagation experiment remains inspectable.
    """

    def __init__(self, layers, seed=42, init_scale=1.0):
        if len(layers) < 2:
            raise ValueError("layers must include input and output sizes")
        self.layers = tuple(int(v) for v in layers)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            limit = init_scale * np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self.history = []

    @property
    def parameter_count(self):
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def _forward(self, x):
        activations = [np.asarray(x, dtype=float)]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.tanh(activations[-1] @ w + b))
        activations.append(activations[-1] @ self.weights[-1] + self.biases[-1])
        return activations

    def predict(self, x):
        return self._forward(x)[-1]

    def _gradients(self, x, y):
        a = self._forward(x)
        # MSE averaged over samples and output coordinates.
        delta = 2.0 * (a[-1] - y) / (y.shape[0] * y.shape[1])
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_w[layer] = a[layer].T @ delta
            grad_b[layer] = delta.sum(axis=0, keepdims=True)
            if layer:
                delta = (delta @ self.weights[layer].T) * (1.0 - a[layer] ** 2)
        return grad_w, grad_b

    @staticmethod
    def mse(y_true, y_pred):
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    @staticmethod
    def good_fraction(y_true, y_pred, tolerance):
        errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        return float(np.mean(np.all(errors <= tolerance, axis=1)))

    def fit(
        self,
        x,
        y,
        x_val=None,
        y_val=None,
        *,
        learning_rate=0.01,
        max_epochs=3000,
        optimizer="adam",
        tolerance=0.05,
        stop_good_fraction=None,
        patience=None,
    ):
        x, y = np.asarray(x, float), np.asarray(y, float)
        mw = [np.zeros_like(w) for w in self.weights]
        vw = [np.zeros_like(w) for w in self.weights]
        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        self.history = []
        best_loss, stale = np.inf, 0

        for epoch in range(1, int(max_epochs) + 1):
            gw, gb = self._gradients(x, y)
            if optimizer == "adam":
                for i in range(len(self.weights)):
                    mw[i] = beta1 * mw[i] + (1 - beta1) * gw[i]
                    vw[i] = beta2 * vw[i] + (1 - beta2) * gw[i] ** 2
                    mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * gb[i] ** 2
                    mwh, vwh = mw[i] / (1 - beta1**epoch), vw[i] / (1 - beta2**epoch)
                    mbh, vbh = mb[i] / (1 - beta1**epoch), vb[i] / (1 - beta2**epoch)
                    self.weights[i] -= learning_rate * mwh / (np.sqrt(vwh) + adam_eps)
                    self.biases[i] -= learning_rate * mbh / (np.sqrt(vbh) + adam_eps)
            elif optimizer == "sgd":
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * gw[i]
                    self.biases[i] -= learning_rate * gb[i]
            else:
                raise ValueError("optimizer must be 'adam' or 'sgd'")

            train_pred = self.predict(x)
            train_loss = self.mse(y, train_pred)
            val_loss = self.mse(y_val, self.predict(x_val)) if x_val is not None else np.nan
            gf = self.good_fraction(y, train_pred, tolerance)
            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 "validation_loss": val_loss, "train_gf": gf})
            monitored = val_loss if np.isfinite(val_loss) else train_loss
            if monitored < best_loss - 1e-12:
                best_loss, stale = monitored, 0
            else:
                stale += 1
            if stop_good_fraction is not None and gf >= stop_good_fraction:
                break
            if patience is not None and stale >= patience:
                break
            if not np.isfinite(train_loss):
                break
        return self

