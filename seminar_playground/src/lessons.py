"""Small pure functions for the first two interactive lessons."""
from __future__ import annotations
import numpy as np

def neuron_metrics(w, b, both=False):
    x = np.array([2., 3.]) if both else np.array([2.]); target = np.array([7., 10.]) if both else np.array([7.])
    prediction = w*x+b; errors = (target-prediction)**2
    return {"x": x, "target": target, "prediction": prediction, "sse": float(errors.sum()), "mse": float(errors.mean())}

def descent_step(w, lr):
    if not np.isfinite(w) or not np.isfinite(lr) or lr <= 0: raise ValueError("w и lr должны быть конечными, lr > 0.")
    grad = 2*(w-2); return {"w": w, "gradient": grad, "lr": lr, "loss": (w-2)**2, "next_w": w-lr*grad}
