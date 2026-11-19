"""Небольшие чистые helpers для предметных Plotly-графиков ЛР7."""
from __future__ import annotations
import numpy as np

def residuals(y_true, y_pred):
    true, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if true.shape != pred.shape:
        raise ValueError("Размеры true/pred не совпадают")
    return true - pred
