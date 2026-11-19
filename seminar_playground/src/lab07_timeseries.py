"""Хронологическая подготовка one-step forecasting без перемешивания."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_lagged(frame, value_column="value", time_column=None, lag=3):
    if not isinstance(lag, (int, np.integer)) or not 1 <= int(lag) <= 64:
        raise ValueError("lag должен быть от 1 до 64")
    if value_column not in frame:
        raise ValueError("Не найден столбец ряда")
    if time_column is not None and time_column not in frame:
        raise ValueError("Не найден столбец времени")
    if len(frame) <= lag:
        raise ValueError("Недостаточно наблюдений для выбранного окна lag")
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Ряд содержит пропуски или Inf")
    if time_column is not None:
        times = frame[time_column].to_numpy()
        if pd.isna(times).any() or pd.Series(times).duplicated().any():
            raise ValueError("Время должно быть непустым и уникальным")
        try:
            if not np.all(times[1:] > times[:-1]):
                raise ValueError("Временной ряд должен быть отсортирован строго по возрастанию")
        except TypeError as exc:
            raise ValueError("Столбец времени должен быть сравнимым") from exc
    else:
        times = np.arange(len(frame))
    rows = []
    for index in range(lag, len(values)):
        row = {f"lag_{step}": values[index - step] for step in range(1, lag + 1)}
        row.update({"target": values[index], "time": times[index]})
        rows.append(row)
    return pd.DataFrame(rows)


def chronological_lag_split(lagged, train=.6, test=.2):
    if not 0 < train < 1 or not 0 < test < 1 or train + test >= 1:
        raise ValueError("Доли train/test должны оставлять непустой control")
    if lagged.empty or not np.isfinite(lagged.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("Лаговая таблица пуста или содержит NaN/Inf")
    a, b = int(len(lagged) * train), int(len(lagged) * (train + test))
    if a < 1 or b <= a or b >= len(lagged):
        raise ValueError("Недостаточно точек для train/test/control")
    parts = {"train": lagged.iloc[:a].reset_index(drop=True),
             "test": lagged.iloc[a:b].reset_index(drop=True),
             "control": lagged.iloc[b:].reset_index(drop=True)}
    if not (parts["train"]["time"].iloc[-1] < parts["test"]["time"].iloc[0] <= parts["test"]["time"].iloc[-1] < parts["control"]["time"].iloc[0]):
        raise ValueError("Границы временных частей нарушают хронологию")
    return parts


def persistence_baseline(frame, lag_columns):
    if not lag_columns or any(column not in frame for column in lag_columns):
        raise ValueError("Не найдены lag-признаки для baseline")
    return frame[lag_columns[0]].to_numpy(float).reshape(-1, 1)
