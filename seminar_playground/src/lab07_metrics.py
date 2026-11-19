"""Метрики ЛР7 с явным смыслом GF/BF."""
from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.shape != pred.shape or true.ndim != 2 or true.shape[0] == 0:
        raise ValueError("y_true и y_pred должны иметь одинаковую непустую форму (N, outputs)")
    if not np.isfinite(true.astype(float)).all() or not np.isfinite(pred.astype(float)).all():
        raise ValueError("Метрики не определяются для NaN/Inf")
    return true.astype(float), pred.astype(float)


def regression_metrics(y_true, y_pred, tolerance=.05):
    true, pred = _arrays(y_true, y_pred)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance должен быть неотрицательным")
    error = true - pred
    abs_error = np.abs(error)
    good = np.all(abs_error <= tolerance, axis=1)
    ss = np.sum((true - true.mean(axis=0)) ** 2)
    return {"mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(abs_error)), "r2": float(1 - np.sum(error ** 2) / ss) if ss > 0 else float("nan"),
            "gf": int(good.sum()), "bf": int((~good).sum()), "gf_fraction": float(good.mean()),
            "max_error": float(abs_error.max()), "mean_error": float(abs_error.mean())}


def confusion_matrix(y_true, y_pred, labels):
    labels = list(labels)
    lookup = {value: index for index, value in enumerate(labels)}
    if len(lookup) != len(labels):
        raise ValueError("Метки классов должны быть уникальными")
    result = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in lookup or predicted not in lookup:
            raise ValueError("В данных встретился неизвестный класс")
        result[lookup[actual], lookup[predicted]] += 1
    return result


def classification_metrics(y_true, y_pred, labels=None):
    true, pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    if true.ndim != 1 or pred.shape != true.shape or len(true) == 0:
        raise ValueError("Метки должны быть непустыми одномерными массивами одинакового размера")
    if labels is None:
        labels = list(dict.fromkeys(true.tolist() + pred.tolist()))
    labels = list(labels)
    matrix = confusion_matrix(true, pred, labels)
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros_like(tp, dtype=float), where=matrix.sum(0) > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros_like(tp, dtype=float), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)
    return {"accuracy": float(np.mean(true == pred)), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "labels": labels,
            "confusion_matrix": matrix.tolist(), "gf": int(np.sum(true == pred)),
            "bf": int(np.sum(true != pred))}


def forecasting_metrics(y_true, y_pred, tolerance=.1):
    result = regression_metrics(y_true, y_pred, tolerance)
    true, pred = _arrays(y_true, y_pred)
    denominator = np.abs(true)
    valid = denominator > 1e-8
    result["mape"] = float(np.mean(np.abs((true[valid] - pred[valid]) / denominator[valid])) * 100) if valid.any() else None
    return result
