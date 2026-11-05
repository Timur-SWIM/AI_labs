"""Deterministic data preparation and experiment records."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .training_engine import TrainingEngine

def coordinate_data(seed=42, count=100):
    if count < 10 or count > 500: raise ValueError("Допустимо от 10 до 500 точек.")
    rng = np.random.default_rng(seed)
    rho, phi = rng.uniform(.05, 1, count), rng.uniform(0, np.pi / 2, count)
    return pd.DataFrame({"rho": rho, "phi": phi, "x": rho*np.cos(phi), "y": rho*np.sin(phi)})

@dataclass
class DataSplit:
    frame: pd.DataFrame; train: pd.DataFrame; validation: pd.DataFrame; control: pd.DataFrame
    mean: np.ndarray; scale: np.ndarray
    def arrays(self, part):
        return ((part[["rho", "phi"]].to_numpy(float)-self.mean)/self.scale, part[["x", "y"]].to_numpy(float))

def make_split(seed=42, count=100, train_size=60, val_size=20):
    if train_size + val_size >= count: raise ValueError("Нужны непустые train, validation и control.")
    frame = coordinate_data(seed, count); order = np.random.default_rng(seed).permutation(count)
    tr, va, co = (frame.iloc[idx].copy().reset_index(names="sample_id") for idx in (order[:train_size], order[train_size:train_size+val_size], order[train_size+val_size:]))
    mean = tr[["rho", "phi"]].to_numpy(float).mean(0); scale = tr[["rho", "phi"]].to_numpy(float).std(0)
    return DataSplit(frame, tr, va, co, mean, np.where(scale == 0, 1, scale))

def evaluate(engine: TrainingEngine, split: DataSplit, include_control=False):
    result = {}
    for name, part in (("train", split.train), ("validation", split.validation)):
        x, y = split.arrays(part); result[name] = engine.metrics(y, engine.predict(x), engine.tolerance)
    if include_control:
        x, y = split.arrays(split.control); result["control"] = engine.metrics(y, engine.predict(x), engine.tolerance)
    return result

def record(engine, metrics):
    return {"architecture": "-".join(map(str, engine.layers)), "optimizer": engine.optimizer, "lr": engine.learning_rate,
            "init_scale": engine.init_scale, "seed": engine.seed, "epochs": engine.epoch, "train_mse": metrics["train"]["mse"],
            "val_mse": metrics["validation"]["mse"], "gf": metrics["validation"]["gf"], "bf": metrics["validation"]["bf"], "history": list(engine.history)}
