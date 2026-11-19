"""Guided UX ЛР7: пять этапов, локальное состояние и реальные вычисления."""
from __future__ import annotations

import io
import json
import hashlib
import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .lab07_research import save_experiment, restore_engine, export_research, import_research, MAX_EXPERIMENTS
from .lab07_datasets import demo_splits
from .lab07_engine import Lab07Engine, one_hot, validate_layers
from .lab07_exports import orange_zip
from .lab07_loader import decode_bytes, detect_delimiter, parse_uploaded, suggest_task, validate_schema
from .lab07_metrics import classification_metrics, forecasting_metrics, regression_metrics
from .lab07_preprocessing import fit_preprocessor
from .lab07_state import short_fingerprint
from .lab07_timeseries import persistence_baseline
from .lab07_ui_helpers import choose_time_value, experiment_fingerprint, model_fingerprint, prepare_timeseries_parts

TASK_NAMES = {"regression": "Аппроксимация", "classification": "Классификация", "timeseries": "Временной ряд"}
STEP_NAMES = ["1. Данные", "2. Постановка", "3. Обучение", "4. Сравнение", "5. Экспорт"]
MISSING = "— отсутствует —"


def _card(task, title, text):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(text)
        st.button("Выбрать", key="lab07_card_" + task, on_click=lambda: st.session_state.update(lab07_task=task, lab07_step=1))


def _demo_preview(task):
    dataset, splits = demo_splits(task, 42)
    st.caption(f"Учебные данные: {dataset.name}. Для своего варианта загрузите файлы.")
    return dataset.frame, splits, {"source": "demo", "name": dataset.name, "seed": dataset.seed, "formula": dataset.formula}


def _parse_uploaded_with_options(uploaded, key):
    raw = uploaded.getvalue()
    text, auto_encoding = decode_bytes(raw)
    st.caption(f"Сырые строки {uploaded.name} (первые 10)")
    st.code("\n".join(text.splitlines()[:10]) or "(пусто)", language="text")
    cols = st.columns(4)
    encoding = cols[0].selectbox("Кодировка", [auto_encoding, "utf-8-sig", "utf-8", "cp1251"], key=key + "encoding")
    detected = detect_delimiter(text)
    delimiter = cols[1].selectbox("Разделитель", [detected, ",", ";", "\t", r"\s+"], key=key + "delimiter")
    decimal = cols[2].selectbox("Десятичный", ["auto", ".", ","], key=key + "decimal")
    has_header = cols[3].toggle("Есть заголовок", value=True, key=key + "header")
    try:
        frame = parse_uploaded(raw, encoding=encoding, delimiter=delimiter, decimal=decimal, header=0 if has_header else None)
    except ValueError as exc:
        st.error(f"{uploaded.name}: {exc}")
        return None, None
    if not has_header:
        frame.columns = [f"feature_{index + 1}" for index in range(frame.shape[1])]
        st.info("Заголовка нет: первая строка сохранена как данные, имена временно назначены feature_1…")
    else:
        renamed = st.text_input("Имена столбцов через запятую (необязательно)", ",".join(map(str, frame.columns)), key=key + "rename")
        names = [item.strip() for item in renamed.split(",") if item.strip()]
        if len(names) == frame.shape[1] and len(set(names)) == len(names):
            frame.columns = names
    return frame, {"name": uploaded.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "encoding": encoding, "delimiter": delimiter, "header": has_header}


def _upload_source(task, key):
    uploads = st.file_uploader("Загрузите Train/Test/Control через браузер", type=["txt", "csv", "tsv"], accept_multiple_files=True, key=key + "files")
    if not uploads:
        st.info("Train обязателен. Test нужен для выбора модели, Control — для однократной финальной оценки.")
        return None, None, None
    parsed, infos = {}, []
    for index, uploaded in enumerate(uploads):
        with st.expander(f"Параметры файла · {uploaded.name}"):
            frame, info = _parse_uploaded_with_options(uploaded, f"{key}_{index}_")
        if frame is not None:
            parsed[uploaded.name] = frame
            infos.append({**info, "rows": len(frame), "columns": len(frame.columns)})
    if not parsed:
        return None, None, None
    with st.expander('Сведения о загруженных файлах'):
        st.dataframe(pd.DataFrame(infos), hide_index=True, use_container_width=True)
    names = list(parsed)
    train_name = st.selectbox("Файл Train", names, index=next((i for i, name in enumerate(names) if "train" in name.lower()), 0), key=key + "train")
    test_name = st.selectbox("Файл Test для выбора модели", [MISSING] + [name for name in names if name != train_name], key=key + "test")
    control_name = st.selectbox("Файл Control для финальной оценки", [MISSING] + [name for name in names if name != train_name and name != test_name], key=key + "control")
    train = parsed[train_name].copy()
    test = None if test_name == MISSING else parsed[test_name].copy()
    control = None if control_name == MISSING else parsed[control_name].copy()
    if test is None:
        if task == "timeseries":
            st.warning("Для временного ряда без Test используется хронологическое разделение, без перемешивания.")
            cut = max(2, int(len(train) * .8)); test = train.iloc[cut:].copy(); train = train.iloc[:cut].copy()
        else:
            rng = np.random.default_rng(42); order = rng.permutation(len(train)); cut = max(1, int(len(train) * .8)); test = train.iloc[order[cut:]].copy(); train = train.iloc[order[:cut]].copy()
        st.warning("Test отсутствовал: создано воспроизводимое разбиение; Control не подменяется.")
    splits = {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True), "control": None if control is None else control.reset_index(drop=True)}
    frame = pd.concat([part for part in splits.values() if part is not None], ignore_index=True)
    return frame, splits, {"source": "upload", "name": train_name, "seed": 42, "files": infos}


def _load_data(task, key):
    source = st.radio("Источник", ["demo", "upload"], format_func=lambda value: "Попробовать на учебных данных" if value == "demo" else "Загрузить свой вариант", horizontal=True, key=key + "source")
    result = _demo_preview(task) if source == "demo" else _upload_source(task, key)
    return (source, *result) if result[0] is not None else (source, None, None, None)


def _plot_source(frame, task, target=None):
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    if task == "classification" and target and len(numeric) >= 2:
        figure = go.Figure()
        for label, part in frame.groupby(target, dropna=False):
            figure.add_scatter(x=part[numeric[0]], y=part[numeric[1]], mode="markers", name=f"Истинный класс: {label}")
        return figure.update_layout(title="Исходные объекты по истинным классам", xaxis_title=numeric[0], yaxis_title=numeric[1])
    if task == "timeseries" and len(frame.columns) >= 2:
        return go.Figure(go.Scatter(x=frame.iloc[:, 0], y=frame.iloc[:, 1], mode="lines+markers", name="ряд")).update_layout(title="Исходный временной ряд", xaxis_title="Время", yaxis_title="Значение")
    if len(numeric) >= 2:
        return go.Figure(go.Scatter(x=frame[numeric[0]], y=frame[numeric[-1]], mode="markers", name="данные")).update_layout(title="Исходные точки")
    return go.Figure().update_layout(title="Недостаточно числовых полей для графика")


def _classification_prediction_plot(raw_test, features, target, predicted, engine=None, preprocessor=None, classes=None):
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(raw_test[column])]
    if len(numeric) < 2:
        return go.Figure().update_layout(title="Недостаточно двух числовых признаков для decision boundary")
    x_col, y_col = numeric[:2]
    figure = go.Figure()
    if engine is not None and preprocessor is not None and classes is not None:
        x_values = raw_test[x_col].astype(float); y_values = raw_test[y_col].astype(float)
        x_grid = np.linspace(float(x_values.min()), float(x_values.max()), 45)
        y_grid = np.linspace(float(y_values.min()), float(y_values.max()), 45)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid = pd.DataFrame({column: raw_test[column].iloc[0] for column in features}, index=range(grid_x.size))
        for column in numeric:
            grid[column] = float(raw_test[column].median())
        grid[x_col] = grid_x.ravel(); grid[y_col] = grid_y.ravel()
        regions = np.argmax(engine.predict(preprocessor.transform(grid)), axis=1).reshape(grid_x.shape)
        figure.add_trace(go.Contour(x=x_grid, y=y_grid, z=regions, contours_coloring="heatmap", opacity=.22, showscale=False, name="Граница решения"))
    true = raw_test[target].astype(str).to_numpy(); predicted = np.asarray(predicted).astype(str)
    for label in sorted(set(true) | set(predicted)):
        mask = true == label
        figure.add_scatter(x=raw_test.loc[mask, x_col], y=raw_test.loc[mask, y_col], mode="markers", name=f"Истинный: {label}", marker_symbol="circle")
    wrong = true != predicted
    if wrong.any():
        figure.add_scatter(x=raw_test.loc[wrong, x_col], y=raw_test.loc[wrong, y_col], mode="markers", name="Ошибки", marker_symbol="x", marker_size=13)
    return figure.update_layout(title="Истинные классы и ошибки предсказания", xaxis_title=x_col, yaxis_title=y_col)


def _primary_plot(task, raw_test, target, prediction, classes=None, baseline=None, features=None, engine=None, preprocessor=None):
    if task == "classification":
        labels = np.asarray([classes[index] for index in np.argmax(prediction, axis=1)], dtype=object)
        return _classification_prediction_plot(raw_test, features or list(raw_test.columns), target, labels, engine, preprocessor, classes), labels
    if task == "timeseries":
        figure = go.Figure([go.Scatter(x=raw_test["time"], y=raw_test[target], mode="lines+markers", name="Факт"), go.Scatter(x=raw_test["time"], y=prediction[:, 0], mode="lines+markers", name="MLP one-step")])
        if baseline is not None: figure.add_scatter(x=raw_test["time"], y=baseline[:, 0], mode="lines", name="Baseline: последнее значение", line=dict(dash="dash"))
        return figure.update_layout(title="Прогноз следующего шага", xaxis_title="Время", yaxis_title="Значение"), None
    figure = go.Figure([go.Scatter(x=np.arange(len(raw_test)), y=raw_test[target], mode="markers", name="Истинное значение"), go.Scatter(x=np.arange(len(raw_test)), y=prediction[:, 0], mode="markers", name="Предсказание")])
    return figure.update_layout(title="Истинные и предсказанные значения", xaxis_title="Объект Test", yaxis_title=target), None


def _prepare_schema(task, frame, splits, key, source):
    if task == "timeseries":
        try: default_value, default_time = choose_time_value(frame)
        except ValueError as exc: st.error(str(exc)); return None
        columns = list(frame.columns)
        value_column = st.selectbox("Value — какое значение прогнозируем", columns, index=columns.index(default_value), key=key + "value")
        time_options = [MISSING] + [column for column in columns if column != value_column]
        time_default = time_options.index(default_time) if default_time in time_options else 0
        time_column = st.selectbox("Time — порядок наблюдений", time_options, index=time_default, key=key + "time")
        if time_column == value_column:
            st.error("Time и Value должны быть разными столбцами."); return None
        lag = st.slider("Lag — сколько прошлых значений подать в сеть", 1, 64, 3, key=key + "lag")
        try:
            prepared = prepare_timeseries_parts(splits, value_column, None if time_column == MISSING else time_column, lag, demo=source == "demo")
        except ValueError as exc:
            st.error(f"Не удалось построить окна: {exc}"); return None
        return [f"lag_{index}" for index in range(1, lag + 1)], "target", None, prepared, time_column if time_column != MISSING else None, value_column, lag
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    defaults = numeric[:min(2, len(numeric))]
    features = st.multiselect("Features — признаки объекта", list(frame.columns), default=defaults, key=key + "features")
    target_options = [column for column in frame.columns if column not in features]
    if not target_options:
        st.error("Оставьте отдельный столбец Target, не включайте цель в Features."); return None
    target = st.selectbox("Target — что сеть должна предсказать", target_options, key=key + "target")
    try: validate_schema(splits["train"], splits.get("test"), splits.get("control"), features, target)
    except ValueError as exc: st.error(str(exc)); return None
    classes = list(dict.fromkeys(splits["train"][target].dropna().tolist())) if task == "classification" else None
    if task == "classification" and len(classes) < 2:
        st.error("В Train должны быть минимум два класса."); return None
    return features, target, classes, splits, None, None, None


def _settings(task, input_dim, output_dim, key):
    preset = st.selectbox("Архитектура", ["простая: 8", "базовая: 8-8", "сложная: 16-16", "ручная"], key=key + "preset")
    if preset.startswith("простая"): hidden = "8"
    elif preset.startswith("базовая"): hidden = "8,8"
    elif preset.startswith("сложная"): hidden = "16,16"
    else: hidden = st.text_input("Скрытые слои через запятую", "8,8", key=key + "hidden")
    try: layers = validate_layers((input_dim, *(int(value.strip()) for value in hidden.split(",") if value.strip()), output_dim), input_dim, output_dim)
    except ValueError as exc: st.error(str(exc)); layers = (input_dim, 8, output_dim)
    columns = st.columns(2)
    learning_rate = float(columns[0].number_input("Learning rate", 1e-5, 2., .01, format="%.5f", key=key + "lr"))
    epochs = int(columns[1].number_input("Эпох за порцию", 1, 5000, 20, key=key + "epochs"))
    with st.expander("Дополнительные параметры модели"):
        seed = int(st.number_input("Seed", 0, 999999, 42, key=key + "seed"))
        optimizer = st.selectbox("Оптимизатор", ["adam", "sgd"], key=key + "optimizer")
        init_scale = float(st.number_input("Масштаб инициализации", .01, 20., 1., key=key + "scale"))
        tolerance = float(st.number_input("Допуск GF/BF", .0001, 100., .1, key=key + "tol"))
    return layers, learning_rate, epochs, seed, optimizer, init_scale, tolerance



def _metrics(task, frame, target, prediction, classes, tolerance):
    if task == 'classification':
        labels = [classes[i] for i in np.argmax(prediction, axis=1)]
        return classification_metrics(frame[target].tolist(), labels, classes)
    y = pd.to_numeric(frame[target], errors='raise').to_numpy(float).reshape(-1, 1)
    return (forecasting_metrics if task == 'timeseries' else regression_metrics)(y, prediction, tolerance)


def _history_table(records, chosen=None):
    rows = []
    for record in records:
        train, test = record['train_metrics'], record['test_metrics']
        score = ('Accuracy', 'Macro-F1') if 'accuracy' in test else ('RMSE', 'MAE')
        keys = ('accuracy', 'macro_f1') if 'accuracy' in test else ('rmse', 'mae')
        row = {'ID': record['id'], 'Модель': record['name'], 'Архитектура': '→'.join(map(str, record['architecture'])),
               'Эпох': record['epochs'], 'Выбрана': record['id'] == chosen}
        row.update({f'Train {label}': train.get(metric) for label, metric in zip(score, keys)})
        row.update({f'Test {label}': test.get(metric) for label, metric in zip(score, keys)})
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_cards(metrics, task, prefix='Test'):
    fields = [('Accuracy', 'accuracy'), ('Macro-F1', 'macro_f1'), ('Верно / ошибок', None)] if task == 'classification' else [('RMSE', 'rmse'), ('MAE', 'mae'), ('Хорошо / плохо', None)]
    for column, (label, field) in zip(st.columns(3), fields):
        value = f"{metrics[field]:.3f}" if field else f"{metrics['gf']} / {metrics['bf']}"
        column.metric(f'{prefix} · {label}', value)


def render_lab07():
    st.title('ЛР7 · Как выбрать модель')
    with st.expander('Дополнительно'):
        teacher = st.toggle('Режим преподавателя', key='lab07_teacher')
    task = st.radio('Выберите маршрут', list(TASK_NAMES), format_func=TASK_NAMES.get, horizontal=True, key='lab07_task')
    key = 'lab07_' + task
    step = int(st.session_state.get(key + 'step', 0))
    st.progress((step + 1) / 3, text=f'{step + 1}/3 · {("Данные и постановка", "Эксперименты", "Выводы и Orange")[step]}')
    if step == 0:
        _stage_data(task, key)
    elif step == 1:
        _stage_experiments(task, key)
    else:
        _stage_final(task, key)
    back, forward = st.columns(2)
    if back.button('Назад', disabled=step == 0, key=key + 'back'):
        st.session_state[key + 'step'] = step - 1; st.rerun()
    records = st.session_state.get(key + 'studies', {}).get(st.session_state.get(key + 'data_fingerprint'), [])
    ready = (step == 0 and key + 'data_fingerprint' in st.session_state) or (step == 1 and (len(records) >= 2 or teacher and len(records) >= 1))
    if forward.button('Далее: ' + ('обучить две модели' if step == 0 else 'выбрать модель для отчёта'), disabled=step == 2 or not ready, key=key + 'next'):
        st.session_state[key + 'step'] = step + 1; st.rerun()
    if step == 1 and len(records) < 2:
        st.caption(f'Сохранено {len(records)} из 2 моделей для перехода к выводам.')


def _stage_data(task, key):
    st.subheader('1/3 · Данные и постановка')
    st.caption('Выберите источник и признаки, затем зафиксируйте постановку.')
    source, frame, splits, meta = _load_data(task, key)
    if frame is None: return
    st.plotly_chart(_plot_source(frame, task), use_container_width=True)
    with st.expander('Показать первые 10 строк'):
        st.dataframe(frame.head(10), use_container_width=True)
    schema = _prepare_schema(task, frame, splits, key, source)
    if schema is None: return
    features, target, classes, prepared, time_column, value_column, lag = schema
    fingerprint = experiment_fingerprint({'train': prepared['train'], 'test': prepared['test'], 'control': prepared.get('control')}, task, features, target,
                                         time_column=time_column, value_column=value_column, lag=lag, preprocessing={'features': features})
    current = st.session_state.get(key + 'data_fingerprint')
    if current and current != fingerprint:
        st.warning('Постановка изменилась. Прежние исследования останутся в архиве по своему fingerprint; модели разных данных нельзя сравнивать.')
    st.caption(f"Train {len(prepared['train'])} · Test {len(prepared['test'])} · Control {len(prepared.get('control')) if prepared.get('control') is not None else 0}")
    with st.expander('Технические сведения'):
        st.code(fingerprint, language='text')
    if st.button('Зафиксировать постановку', key=key + 'commit_data', type='primary'):
        st.session_state[key + 'data'] = (source, frame, splits, meta, schema)
        st.session_state[key + 'data_fingerprint'] = fingerprint
        st.session_state.pop(key + 'engine', None)
        st.session_state.pop(key + 'model_fingerprint', None)
        st.session_state.pop(key + 'locked_id', None)
        st.session_state.pop(key + 'control_result', None)
        st.session_state.setdefault(key + 'studies', {}).setdefault(fingerprint, [])
        st.success('Постановка зафиксирована. Далее обучите две модели.')


def _context(task, key):
    data = st.session_state.get(key + 'data')
    if data is None:
        st.warning('Сначала зафиксируйте постановку на этапе 1.'); return None
    source, frame, splits, meta, schema = data
    features, target, classes, prepared, time_column, value_column, lag = schema
    preprocessor = fit_preprocessor(prepared['train'][features], features)
    x_train = preprocessor.transform(prepared['train'][features]); x_test = preprocessor.transform(prepared['test'][features])
    if task == 'classification':
        y_train = one_hot(prepared['train'][target].tolist(), classes); y_test = one_hot(prepared['test'][target].tolist(), classes)
    else:
        y_train = pd.to_numeric(prepared['train'][target], errors='raise').to_numpy(float).reshape(-1, 1)
        y_test = pd.to_numeric(prepared['test'][target], errors='raise').to_numpy(float).reshape(-1, 1)
    return locals()


def _stage_experiments(task, key):
    st.subheader('2/3 · Эксперименты')
    ctx = _context(task, key)
    if ctx is None: return
    fp = st.session_state[key + 'data_fingerprint']
    records = st.session_state.setdefault(key + 'studies', {}).setdefault(fp, [])
    st.caption(f"Обучите и сохраните две модели, затем сравните Test. {ctx['meta']['name']} · Train {len(ctx['prepared']['train'])} · Test {len(ctx['prepared']['test'])}.")
    left, main = st.columns([1, 2])
    with left:
        st.markdown('#### Рабочая модель')
        settings = _settings(task, ctx['x_train'].shape[1], ctx['y_train'].shape[1], key)
        layers, lr, epochs, seed, optimizer, scale, tolerance = settings
        model_fp = model_fingerprint(data_fingerprint=fp, architecture=layers, optimizer=optimizer, learning_rate=lr, seed=seed, init_scale=scale, task=task)
        engine = st.session_state.get(key + 'engine')
        if engine is not None and st.session_state.get(key + 'model_fingerprint') != model_fp:
            st.warning('Параметры изменились. Для новых настроек запустите обучение заново; сохранённые модели остаются в истории.')
            engine = None
        st.caption('MLP: ' + ' → '.join(map(str, layers)))
        if st.button('Запустить обучение', key=key + 'start', type='primary'):
            engine = Lab07Engine(layers, task, seed, init_scale=scale, optimizer=optimizer, learning_rate=lr, tolerance=tolerance)
            engine.train_epochs(epochs, ctx['x_train'], ctx['y_train'], ctx['x_test'], ctx['y_test'])
            st.session_state[key + 'engine'] = engine; st.session_state[key + 'model_fingerprint'] = model_fp; st.rerun()
        if st.button(f'Продолжить +{epochs} эпох', disabled=engine is None, key=key + 'continue'):
            engine.stopped = False; engine.train_epochs(epochs, ctx['x_train'], ctx['y_train'], ctx['x_test'], ctx['y_test']); st.rerun()
        if st.button('Новый эксперимент', key=key + 'new'):
            st.session_state.pop(key + 'engine', None); st.session_state.pop(key + 'model_fingerprint', None); st.rerun()
        if engine is not None:
            name = st.text_input('Имя модели', value=chr(65 + len(records)) if len(records) < 26 else f'Модель {len(records)+1}', key=key + 'experiment_name')
            if st.button('Сохранить эксперимент', key=key + 'save'):
                prediction = engine.predict(ctx['x_test'])
                try:
                    save_experiment(records, data_fingerprint=fp, model_fingerprint=model_fp, engine=engine,
                                    train_metrics=_metrics(task, ctx['prepared']['train'], ctx['target'], engine.predict(ctx['x_train']), ctx['classes'], tolerance),
                                    test_metrics=_metrics(task, ctx['prepared']['test'], ctx['target'], prediction, ctx['classes'], tolerance),
                                    name=name, predictions=prediction)
                except ValueError as exc: st.error(str(exc))
                else: st.rerun()
    with main:
        if engine is None: st.caption('График и метрики появятся после обучения.')
        else:
            pred = engine.predict(ctx['x_test'])
            baseline = persistence_baseline(ctx['prepared']['test'], ['lag_1']) if task == 'timeseries' else None
            fig, _ = _primary_plot(task, ctx['prepared']['test'], ctx['target'], pred, ctx['classes'], baseline=baseline,
                                   features=ctx['features'], engine=engine, preprocessor=ctx['preprocessor'])
            st.plotly_chart(fig, use_container_width=True)
            metrics = _metrics(task, ctx['prepared']['test'], ctx['target'], pred, ctx['classes'], tolerance)
            _metric_cards(metrics, task)
            with st.expander('Кривые Loss Train/Test'):
                st.line_chart(pd.DataFrame(engine.history).set_index('epoch')[['train_loss', 'validation_loss']])
    st.markdown('#### Сохранённые эксперименты')
    if records: st.dataframe(_history_table(records, st.session_state.get(key + 'locked_id')), hide_index=True, use_container_width=True)
    else: st.caption('Сохранённых моделей пока нет.')
    if records:
        ids = [r['id'] for r in records]
        compare = st.multiselect('Модели для сравнения', ids, default=ids[:2], key=key + 'compare_ids')
        if len(compare) >= 2:
            fig = go.Figure()
            predictions_figure = go.Figure()
            for record in records:
                if record['id'] in compare:
                    h = record['snapshot']['history']
                    fig.add_scatter(x=[row['epoch'] for row in h], y=[row['validation_loss'] for row in h], name=record['name'] + ' Test')
                    values = np.asarray(record['predictions'])
                    predictions_figure.add_scatter(x=list(range(len(values))), y=np.argmax(values, axis=1) if task == 'classification' else values[:, 0], name=record['name'])
            st.plotly_chart(fig.update_layout(title='Test Loss по фактическим эпохам'), use_container_width=True)
            st.plotly_chart(predictions_figure.update_layout(title='Предсказания на одном Test'), use_container_width=True)
    with st.expander('Управление историей'):
        st.caption('История хранится в текущей сессии. Скачать её можно в ZIP на последнем этапе.')
        upload = st.file_uploader('Восстановить исследование из JSON (до 8 МБ)', type=['json'], key=key + 'research_import')
        if upload and st.button('Восстановить историю', key=key + 'import'):
            try: imported = import_research(upload.getvalue(), fp)
            except (ValueError, KeyError, TypeError) as exc: st.error(f'Импорт отклонён: {exc}')
            else:
                existing = {r['id'] for r in records}
                records.extend(r for r in imported if r['id'] not in existing)
                st.rerun()
        if records and st.checkbox('Подтверждаю очистку истории текущей постановки', key=key + 'clear_confirm') and st.button('Очистить историю', key=key + 'clear'):
            records.clear(); st.session_state.pop(key + 'locked_id', None); st.session_state.pop(key + 'control_result', None); st.rerun()

def _stage_final(task, key):
    st.subheader('3/3 · Выводы и Orange')
    ctx = _context(task, key)
    if ctx is None: return
    fp = st.session_state[key + 'data_fingerprint']
    records = st.session_state.get(key + 'studies', {}).get(fp, [])
    if not records: st.warning('Сначала сохраните модель на этапе 2.'); return
    st.caption('Сравните Test, выберите модель и зафиксируйте её до проверки Control.')
    st.dataframe(_history_table(records, st.session_state.get(key + 'locked_id')), hide_index=True, use_container_width=True)
    ids = [r['id'] for r in records]
    chosen = st.selectbox('Сохранённая модель для отчёта', ids, format_func=lambda rid: next(f"{r['name']} ({rid}) · Test {r['test_metrics'].get('rmse', r['test_metrics'].get('accuracy')):.3f}" for r in records if r['id'] == rid), key=key + 'chosen_id')
    record = next(r for r in records if r['id'] == chosen)
    if record['data_fingerprint'] != fp: st.error('Модель принадлежит другой постановке.'); return
    locked_id = st.session_state.get(key + 'locked_id')
    if st.button('Зафиксировать выбор', disabled=locked_id is not None, key=key + 'lock', type='primary'):
        st.session_state[key + 'locked_id'] = chosen; locked_id = chosen
    if locked_id is None:
        st.caption('После фиксации выбора станет доступна итоговая проверка Control.')
        return
    record = next(r for r in records if r['id'] == locked_id)
    frozen = restore_engine(record['snapshot'])
    st.caption(f"Зафиксирована модель {record['name']} ({locked_id}), эпоха {record['epochs']}.")
    _metric_cards(record['test_metrics'], task)
    control = ctx['prepared'].get('control')
    result = st.session_state.get(key + 'control_result')
    if control is not None and result is None:
        if st.button('Итоговая проверка Control', key=key + 'control'):
            prediction = frozen.predict(ctx['preprocessor'].transform(control[ctx['features']]))
            result = {'id': locked_id, 'data_fingerprint': fp, 'metrics': _metrics(task, control, ctx['target'], prediction, ctx['classes'], frozen.tolerance)}
            st.session_state[key + 'control_result'] = result; st.rerun()
    if result is not None and result['id'] == locked_id and result['data_fingerprint'] == fp:
        st.markdown(f"#### Control · модель {record['name']} ({locked_id})")
        _metric_cards(result['metrics'], task, 'Control')
        if task == 'classification':
            with st.expander('Матрица ошибок'):
                st.dataframe(pd.DataFrame(result['metrics']['confusion_matrix'], index=ctx['classes'], columns=ctx['classes']))
    elif control is None: st.info('Control не загружен; независимая проверка недоступна.')
    if control is not None and result is None:
        return
    prediction = frozen.predict(ctx['x_test'])
    test_predictions = ctx['prepared']['test'].copy()
    test_predictions['prediction'] = [ctx['classes'][i] for i in np.argmax(prediction, axis=1)] if task == 'classification' else prediction[:, 0]
    control_predictions = None
    if result is not None and control is not None:
        cp = frozen.predict(ctx['preprocessor'].transform(control[ctx['features']]))
        control_predictions = control.copy()
        control_predictions['prediction'] = [ctx['classes'][i] for i in np.argmax(cp, axis=1)] if task == 'classification' else cp[:, 0]
    manifest = {'source': ctx['meta']['source'], 'task': task, 'model_id': locked_id, 'model_name': record['name'],
                'data_fingerprint': fp, 'model_fingerprint': record['model_fingerprint'], 'architecture': record['architecture'],
                'optimizer': record['optimizer'], 'learning_rate': record['learning_rate'], 'seed': record['seed'],
                'epochs': record['epochs'], 'features': ctx['features'], 'target': ctx['target'], 'time': ctx['time_column'],
                'value': ctx['value_column'], 'lag': ctx['lag'], 'preprocessing': 'fit on Train only',
                'control_revealed': result is not None, 'limitations': 'Results are session-local; Orange retrains its own model.'}
    package = orange_zip(dataset_manifest=manifest, train=ctx['prepared']['train'], test=ctx['prepared']['test'], control=control,
                         predictions=test_predictions, control_predictions=control_predictions,
                         experiment_config={k: manifest[k] for k in ('model_id', 'architecture', 'optimizer', 'learning_rate', 'seed', 'epochs', 'data_fingerprint')},
                         metrics={'test': record['test_metrics'], **({'control': result['metrics']} if result else {})},
                         experiments=_history_table(records, locked_id), research_json=export_research(fp, records),
                         report=f"# ЛР7 · модель {record['name']} ({locked_id})\n\nВыбрана по Test и зафиксирована до Control.\n")
    st.markdown('#### Результаты и Orange')
    st.caption('ZIP содержит таблицу сравнения, предсказания, CSV для Orange, JSON исследования и инструкцию. Сохраните его: история в приложении живёт только в текущей сессии.')
    st.download_button('Скачать результаты ZIP', package, 'lab07_results.zip', 'application/zip', key=key + 'download', type='primary')
