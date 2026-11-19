"""Воспроизводимые демонстрационные наборы ЛР7; это не данные вариантов."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DemoDataset:
    name: str
    task: str
    frame: pd.DataFrame
    formula: str
    seed: int
    source: str = "demo"

def _split(frame: pd.DataFrame, seed: int, train=.6, test=.2):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    a, b = int(len(frame)*train), int(len(frame)*(train+test))
    return {"train": frame.iloc[order[:a]].reset_index(drop=True),
            "test": frame.iloc[order[a:b]].reset_index(drop=True),
            "control": frame.iloc[order[b:]].reset_index(drop=True)}

def regression_demo(seed=42, count=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.2, 2.2, (count, 2))
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + .12*x[:, 0]**2 - .08*x[:, 1]
    frame = pd.DataFrame({"x1": x[:,0], "x2": x[:,1], "target": y})
    return DemoDataset("Гладкая поверхность", "regression", frame,
        "target = sin(x1) cos(x2) + 0.12 x1² - 0.08 x2", seed)

def classification_demo(seed=42, count=240):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.4, 2.4, (count, 2))
    radius = np.sqrt((x*x).sum(1))
    label = np.where((radius + .28*np.sin(3*x[:,0]) < 1.35), "внутренний", "внешний")
    frame = pd.DataFrame({"x1": x[:,0], "x2": x[:,1], "class": label})
    return DemoDataset("Нелинейные кольца", "classification", frame,
        "class = внутренний, если sqrt(x1²+x2²)+0.28 sin(3x1) < 1.35", seed)

def timeseries_demo(seed=42, count=240):
    rng = np.random.default_rng(seed)
    t = np.arange(count)
    value = .018*t + .65*np.sin(2*np.pi*t/24) + .22*np.sin(2*np.pi*t/7) + rng.normal(0, .08, count)
    frame = pd.DataFrame({"time": t, "value": value})
    return DemoDataset("Тренд и сезонность", "timeseries", frame,
        "value(t)=0.018t+0.65 sin(2πt/24)+0.22 sin(2πt/7)+N(0,0.08²)", seed)

def demo_splits(task: str, seed=42):
    dataset = {"regression": regression_demo, "classification": classification_demo,
               "timeseries": timeseries_demo}[task](seed)
    return dataset, _split(dataset.frame, seed) if task != "timeseries" else chronological_split(dataset.frame)

def chronological_split(frame, train=.6, test=.2):
    a, b = int(len(frame)*train), int(len(frame)*(train+test))
    return {"train": frame.iloc[:a].reset_index(drop=True),
            "test": frame.iloc[a:b].reset_index(drop=True),
            "control": frame.iloc[b:].reset_index(drop=True)}
