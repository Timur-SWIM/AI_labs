"""Чистые helpers для guided UI и интеграционных тестов."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .lab07_state import dataset_fingerprint
from .lab07_timeseries import chronological_lag_split, make_lagged, make_lagged_parts


def choose_time_value(frame: pd.DataFrame) -> tuple[str, str | None]:
    if "value" in frame.columns:
        value = "value"
    else:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            raise ValueError("Не найден числовой столбец значения ряда")
        value = numeric[-1]
    if "time" in frame.columns:
        time = "time"
    else:
        candidates = [column for column in frame.columns if str(column).lower() in {"date", "timestamp", "datetime", "index"}]
        time = candidates[0] if candidates else None
    if time == value:
        raise ValueError("Столбец времени не может одновременно быть значением ряда")
    return value, time


def prepare_timeseries_parts(parts: dict[str, pd.DataFrame | None], value_column: str,
                             time_column: str | None, lag: int, demo: bool = False):
    clean = {name: frame for name, frame in parts.items() if frame is not None}
    if demo:
        full = pd.concat([clean[name] for name in ("train", "test", "control") if name in clean], ignore_index=True)
        lagged = make_lagged(full, value_column, time_column, lag)
        return chronological_lag_split(lagged)
    return make_lagged_parts(clean, value_column, time_column, lag)


def experiment_fingerprint(parts, task, features, target, *, time_column=None, value_column=None, lag=None,
                           preprocessing=None, model=None):
    return dataset_fingerprint(parts=parts, task=task, features=features, target=target,
                               time_column=time_column, value_column=value_column, lag=lag,
                               preprocessing=preprocessing, model=model)


def model_fingerprint(*, data_fingerprint: str, architecture, optimizer: str, learning_rate: float,
                      seed: int, init_scale: float, task: str) -> str:
    payload = {"data": data_fingerprint, "architecture": list(architecture), "optimizer": optimizer,
               "learning_rate": learning_rate, "seed": seed, "init_scale": init_scale, "task": task}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
