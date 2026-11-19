"""Independent experiment snapshots and bounded, local research exchange."""
from __future__ import annotations

import copy
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np

from .lab07_engine import Lab07Engine

MAX_EXPERIMENTS = 20
MAX_IMPORT_BYTES = 8_000_000


def restore_engine(snapshot):
    engine = Lab07Engine(tuple(snapshot['layers']), snapshot['task'], int(snapshot['seed']),
                         init_scale=float(snapshot['init_scale']), optimizer=snapshot['optimizer'],
                         learning_rate=float(snapshot['learning_rate']), tolerance=float(snapshot['tolerance']))
    for field in ('weights', 'biases', 'mw', 'vw', 'mb', 'vb'):
        values = [np.asarray(value, dtype=float) for value in snapshot[field]]
        template = getattr(engine, field)
        if len(values) != len(template) or any(a.shape != b.shape or not np.isfinite(a).all() for a, b in zip(values, template)):
            raise ValueError(f'Некорректный snapshot: {field}')
        setattr(engine, field, [value.copy() for value in values])
    engine.epoch = int(snapshot['epoch'])
    engine.stopped = bool(snapshot['stopped'])
    engine.history = copy.deepcopy(snapshot['history'])
    return engine


def save_experiment(history, *, data_fingerprint, model_fingerprint, engine, train_metrics, test_metrics, name, predictions):
    if len(history) >= MAX_EXPERIMENTS:
        raise ValueError(f'Лимит: {MAX_EXPERIMENTS} моделей. Экспортируйте исследование перед очисткой истории.')
    record = {'id': f'E{len(history) + 1:03d}', 'name': name,
              'timestamp': datetime.now(timezone.utc).isoformat(),
              'data_fingerprint': data_fingerprint, 'model_fingerprint': model_fingerprint,
              'architecture': list(engine.layers), 'parameters': engine.parameter_count,
              'optimizer': engine.optimizer, 'learning_rate': engine.learning_rate,
              'seed': engine.seed, 'epochs': engine.epoch,
              'train_metrics': copy.deepcopy(train_metrics), 'test_metrics': copy.deepcopy(test_metrics),
              'predictions': np.asarray(predictions).tolist(), 'snapshot': copy.deepcopy(engine.snapshot())}
    history.append(record)
    return record


def export_research(data_fingerprint, records):
    def default(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    def clean(value):
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, np.generic): return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value): return None
        if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [clean(item) for item in value]
        return value
    payload = clean({'schema_version': 1, 'data_fingerprint': data_fingerprint, 'experiments': records})
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, default=default).encode('utf-8')


def import_research(raw, expected_fingerprint):
    if len(raw) > MAX_IMPORT_BYTES: raise ValueError('Файл исследования больше 8 МБ')
    payload = json.loads(raw)
    if payload.get('schema_version') != 1 or payload.get('data_fingerprint') != expected_fingerprint:
        raise ValueError('Формат или данные исследования не совпадают с текущей постановкой')
    records = payload.get('experiments')
    if not isinstance(records, list) or len(records) > MAX_EXPERIMENTS: raise ValueError('Неверное число экспериментов')
    for record in records:
        if record.get('data_fingerprint') != expected_fingerprint: raise ValueError('Смешаны разные наборы данных')
        restore_engine(record['snapshot'])
        for field in ('id', 'name', 'model_fingerprint', 'test_metrics', 'train_metrics', 'predictions'):
            if field not in record: raise ValueError(f'Отсутствует поле {field}')
    return copy.deepcopy(records)
