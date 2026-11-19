"""Предобработка, обученная только по Train."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TrainPreprocessor:
    feature_columns: list[str]
    categorical: list[str]
    means: dict[str, float]
    scales: dict[str, float]
    categories: dict[str, list[str]]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if any(column not in frame.columns for column in self.feature_columns):
            raise ValueError("В данных отсутствует выбранный признак")
        blocks: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.categorical:
                known = self.categories[column]
                values = frame[column].fillna("<пропуск>").astype(str)
                blocks.append(np.column_stack([(values == category).to_numpy(float) for category in known]))
            else:
                values = pd.to_numeric(frame[column], errors="coerce")
                values = values.replace([np.inf, -np.inf], np.nan).fillna(self.means[column]).to_numpy(float)
                blocks.append(((values - self.means[column]) / self.scales[column]).reshape(-1, 1))
        return np.concatenate(blocks, axis=1) if blocks else np.empty((len(frame), 0))

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame)


def fit_preprocessor(train: pd.DataFrame, feature_columns: list[str]) -> TrainPreprocessor:
    if train.empty:
        raise ValueError("Нельзя обучить препроцессор на пустом Train")
    categorical: list[str] = []
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    categories: dict[str, list[str]] = {}
    for column in feature_columns:
        if column not in train:
            raise ValueError(f"Признак {column!r} отсутствует")
        numeric = pd.to_numeric(train[column], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        # Строковые признаки и смешанные колонки кодируются one-hot; неизвестные Test/Control = all-zero.
        if len(finite) >= max(1, int(0.8 * len(train))):
            means[column] = float(np.mean(finite)) if len(finite) else 0.0
            scale = float(np.std(finite)) if len(finite) else 1.0
            scales[column] = scale if scale > 1e-12 else 1.0
        else:
            categorical.append(column)
            values = train[column].fillna("<пропуск>").astype(str)
            categories[column] = sorted(set(values.tolist())) or ["<пропуск>"]
    return TrainPreprocessor(list(feature_columns), categorical, means, scales, categories)
