"""Stateful NumPy MLP used by the playground.

The forward/backward formulae originate from ``lab06/src/mlp.py`` in this
repository (MLPRegressor).  This standalone copy adds a serialisable optimizer
state and epoch-at-a-time API; it deliberately keeps tanh hidden layers, a
linear output layer and mean-squared error unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

MAX_HIDDEN_LAYERS, MAX_NEURONS, MAX_EPOCHS = 4, 32, 5000


def validate_layers(layers: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in layers)
    if len(layers) < 2 or layers[0] != 2 or layers[-1] != 2:
        raise ValueError("Архитектура должна начинаться и заканчиваться двумя нейронами.")
    if len(layers) - 2 > MAX_HIDDEN_LAYERS or any(v < 1 or v > MAX_NEURONS for v in layers[1:-1]):
        raise ValueError(f"Допустимо до {MAX_HIDDEN_LAYERS} скрытых слоёв по 1–{MAX_NEURONS} нейрона.")
    return layers


@dataclass
class TrainingEngine:
    layers: tuple[int, ...]
    seed: int = 42
    init_scale: float = 1.0
    optimizer: str = "adam"
    learning_rate: float = 0.01
    tolerance: float = 0.05
    weights: list[np.ndarray] = field(init=False)
    biases: list[np.ndarray] = field(init=False)
    mw: list[np.ndarray] = field(init=False)
    vw: list[np.ndarray] = field(init=False)
    mb: list[np.ndarray] = field(init=False)
    vb: list[np.ndarray] = field(init=False)
    epoch: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self):
        self.layers = validate_layers(self.layers)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate должен быть положительным.")
        if not np.isfinite(self.init_scale) or self.init_scale <= 0:
            raise ValueError("init_scale должен быть положительным.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer должен быть 'adam' или 'sgd'.")
        rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            limit = self.init_scale * np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        a = [np.asarray(x, dtype=float)]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a.append(np.tanh(a[-1] @ w + b))
        a.append(a[-1] @ self.weights[-1] + self.biases[-1])
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[-1]

    def gradients(self, x: np.ndarray, y: np.ndarray):
        y = np.asarray(y, float); a = self._forward(x)
        delta = 2.0 * (a[-1] - y) / (y.shape[0] * y.shape[1])
        gw, gb = [None] * len(self.weights), [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            gw[layer] = a[layer].T @ delta
            gb[layer] = delta.sum(axis=0, keepdims=True)
            if layer:
                delta = (delta @ self.weights[layer].T) * (1.0 - a[layer] ** 2)
        return gw, gb

    @staticmethod
    def mse(y_true, y_pred) -> float:
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    @staticmethod
    def metrics(y_true, y_pred, tolerance: float) -> dict:
        errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        good = np.all(errors <= tolerance, axis=1)
        return {"mse": float(np.mean(errors**2)), "mae": float(errors.mean()),
                "gf": int(good.sum()), "bf": int((~good).sum()), "gf_fraction": float(good.mean())}

    def train_epoch(self, x_train, y_train, x_val=None, y_val=None) -> dict:
        if self.stopped:
            return self.history[-1] if self.history else {}
        gw, gb = self.gradients(x_train, y_train)
        next_epoch = self.epoch + 1
        if self.optimizer == "adam":
            for i in range(len(self.weights)):
                self.mw[i] = .9 * self.mw[i] + .1 * gw[i]; self.vw[i] = .999 * self.vw[i] + .001 * (gw[i] ** 2)
                self.mb[i] = .9 * self.mb[i] + .1 * gb[i]; self.vb[i] = .999 * self.vb[i] + .001 * (gb[i] ** 2)
                self.weights[i] -= self.learning_rate * (self.mw[i] / (1 - .9**next_epoch)) / (np.sqrt(self.vw[i] / (1 - .999**next_epoch)) + 1e-8)
                self.biases[i] -= self.learning_rate * (self.mb[i] / (1 - .9**next_epoch)) / (np.sqrt(self.vb[i] / (1 - .999**next_epoch)) + 1e-8)
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gw[i]; self.biases[i] -= self.learning_rate * gb[i]
        self.epoch = next_epoch
        train = self.metrics(y_train, self.predict(x_train), self.tolerance)
        val_mse = np.nan if x_val is None else self.mse(y_val, self.predict(x_val))
        row = {"epoch": self.epoch, "train_loss": train["mse"], "validation_loss": val_mse, "train_gf": train["gf_fraction"]}
        self.history.append(row)
        if not np.isfinite(row["train_loss"]): self.stopped = True
        return row

    def train_epochs(self, count, x_train, y_train, x_val=None, y_val=None, stop_good_fraction=None):
        if not isinstance(count, (int, np.integer)) or count < 0 or count > MAX_EPOCHS:
            raise ValueError(f"Количество эпох должно быть от 0 до {MAX_EPOCHS}.")
        for _ in range(count):
            row = self.train_epoch(x_train, y_train, x_val, y_val)
            if self.stopped or (stop_good_fraction is not None and row["train_gf"] >= stop_good_fraction): break
        return self.history

    def snapshot(self) -> dict:
        return {"layers": self.layers, "seed": self.seed, "init_scale": self.init_scale, "optimizer": self.optimizer,
                "learning_rate": self.learning_rate, "tolerance": self.tolerance, "epoch": self.epoch,
                "weights": [w.copy() for w in self.weights], "biases": [b.copy() for b in self.biases],
                "mw": [v.copy() for v in self.mw], "vw": [v.copy() for v in self.vw], "mb": [v.copy() for v in self.mb], "vb": [v.copy() for v in self.vb], "history": list(self.history)}
