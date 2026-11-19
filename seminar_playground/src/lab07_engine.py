"""Универсальный обозримый NumPy MLP для ЛР7; движок ЛР6 не изменяется."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

MAX_HIDDEN_LAYERS, MAX_NEURONS, MAX_EPOCHS = 4, 32, 5000


def validate_layers(layers, input_dim=None, output_dim=None):
    values = tuple(int(value) for value in layers)
    if len(values) < 2 or len(values) - 2 > MAX_HIDDEN_LAYERS:
        raise ValueError("Допустимо от 0 до 4 скрытых слоёв")
    if any(value < 1 or value > MAX_NEURONS for value in values[1:-1]):
        raise ValueError("Скрытые слои: от 1 до 32 нейронов")
    if input_dim is not None and values[0] != int(input_dim):
        raise ValueError("Размер входа не совпадает с числом признаков после кодирования")
    if output_dim is not None and values[-1] != int(output_dim):
        raise ValueError("Размер выхода не совпадает с целью")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, float) - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def one_hot(labels, classes) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes)}
    result = np.zeros((len(labels), len(classes)), dtype=float)
    for row, label in enumerate(labels):
        if label not in lookup:
            raise ValueError(f"Неизвестный класс {label!r}; классы должны обучаться по Train")
        result[row, lookup[label]] = 1.0
    return result


@dataclass
class Lab07Engine:
    layers: tuple[int, ...]
    task: str = "regression"
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
        if self.task not in {"regression", "classification", "timeseries"}:
            raise ValueError("task должен быть regression, timeseries или classification")
        # Forecasting использует тот же линейный выход/MSE, но сохраняет семантику режима.
        if self.task == "timeseries":
            self.task = "timeseries"
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer должен быть adam или sgd")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным")
        if not np.isfinite(self.init_scale) or self.init_scale <= 0:
            raise ValueError("init_scale должен быть положительным")
        rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            limit = self.init_scale * np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))
        self.mw = [np.zeros_like(weight) for weight in self.weights]
        self.vw = [np.zeros_like(weight) for weight in self.weights]
        self.mb = [np.zeros_like(bias) for bias in self.biases]
        self.vb = [np.zeros_like(bias) for bias in self.biases]

    @property
    def parameter_count(self) -> int:
        return int(sum((fan_in + 1) * fan_out for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:])))

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.layers[0]:
            raise ValueError(f"Ожидался массив формы (N, {self.layers[0]})")
        if not np.isfinite(x).all():
            raise ValueError("Признаки содержат NaN или Inf")
        activations = [x]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.tanh(activations[-1] @ weight + bias))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        activations.append(softmax(logits) if self.task == "classification" else logits)
        return activations

    def predict(self, x):
        return self._forward(x)[-1]

    def loss(self, y, prediction) -> float:
        y, prediction = np.asarray(y, float), np.asarray(prediction, float)
        if y.shape != prediction.shape:
            raise ValueError("Размеры target и prediction не совпадают")
        if self.task == "classification":
            return float(-np.mean(np.sum(y * np.log(np.clip(prediction, 1e-12, 1.0)), axis=1)))
        return float(np.mean((prediction - y) ** 2))

    def gradients(self, x, y):
        y = np.asarray(y, dtype=float)
        activations = self._forward(x)
        if y.ndim != 2 or y.shape != activations[-1].shape or not np.isfinite(y).all():
            raise ValueError("Target имеет неверную форму или содержит NaN/Inf")
        n = y.shape[0]
        if self.task == "classification":
            delta = (activations[-1] - y) / n
        else:
            delta = 2.0 * (activations[-1] - y) / (n * y.shape[1])
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        for index in range(len(self.weights) - 1, -1, -1):
            grad_w[index] = activations[index].T @ delta
            grad_b[index] = delta.sum(axis=0, keepdims=True)
            if index:
                delta = (delta @ self.weights[index].T) * (1.0 - activations[index] ** 2)
        return grad_w, grad_b

    def train_epoch(self, x, y, x_val=None, y_val=None):
        if self.stopped:
            return self.history[-1] if self.history else {}
        grad_w, grad_b = self.gradients(x, y)
        step = self.epoch + 1
        for index in range(len(self.weights)):
            if self.optimizer == "adam":
                self.mw[index] = .9 * self.mw[index] + .1 * grad_w[index]
                self.vw[index] = .999 * self.vw[index] + .001 * grad_w[index] ** 2
                self.mb[index] = .9 * self.mb[index] + .1 * grad_b[index]
                self.vb[index] = .999 * self.vb[index] + .001 * grad_b[index] ** 2
                mw = self.mw[index] / (1 - .9 ** step); vw = self.vw[index] / (1 - .999 ** step)
                mb = self.mb[index] / (1 - .9 ** step); vb = self.vb[index] / (1 - .999 ** step)
                self.weights[index] -= self.learning_rate * mw / (np.sqrt(vw) + 1e-8)
                self.biases[index] -= self.learning_rate * mb / (np.sqrt(vb) + 1e-8)
            else:
                self.weights[index] -= self.learning_rate * grad_w[index]
                self.biases[index] -= self.learning_rate * grad_b[index]
        self.epoch = step
        prediction = self.predict(x)
        row = {"epoch": step, "train_loss": self.loss(y, prediction),
               "validation_loss": np.nan if x_val is None else self.loss(y_val, self.predict(x_val))}
        if self.task == "classification":
            row["train_accuracy"] = float(np.mean(np.argmax(prediction, 1) == np.argmax(y, 1)))
            if x_val is not None:
                row["validation_accuracy"] = float(np.mean(np.argmax(self.predict(x_val), 1) == np.argmax(y_val, 1)))
        self.history.append(row)
        if not np.isfinite(row["train_loss"]) or any(not np.isfinite(value).all() for value in self.weights + self.biases):
            self.stopped = True
        return row

    def train_epochs(self, count, x, y, x_val=None, y_val=None):
        if not isinstance(count, (int, np.integer)) or not 0 <= count <= MAX_EPOCHS:
            raise ValueError(f"Количество эпох должно быть от 0 до {MAX_EPOCHS}")
        for _ in range(int(count)):
            self.train_epoch(x, y, x_val, y_val)
            if self.stopped:
                break
        return self.history

    def snapshot(self):
        return {"layers": self.layers, "task": self.task, "seed": self.seed,
                "init_scale": self.init_scale, "optimizer": self.optimizer,
                "learning_rate": self.learning_rate, "tolerance": self.tolerance,
                "epoch": self.epoch, "stopped": self.stopped,
                "weights": [value.copy() for value in self.weights],
                "biases": [value.copy() for value in self.biases],
                "mw": [value.copy() for value in self.mw], "vw": [value.copy() for value in self.vw],
                "mb": [value.copy() for value in self.mb], "vb": [value.copy() for value in self.vb],
                "history": list(self.history)}
