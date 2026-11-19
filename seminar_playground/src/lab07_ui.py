"""Guided UX ЛР7: пять этапов, локальное состояние и реальные вычисления."""
from __future__ import annotations

import io
import json
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    st.warning("Демонстрационные данные — не данные индивидуального варианта.")
    st.caption(f"Набор: {dataset.name} · seed={dataset.seed} · формула: {dataset.formula}")
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
        frame, info = _parse_uploaded_with_options(uploaded, f"{key}_{index}_")
        if frame is not None:
            parsed[uploaded.name] = frame
            infos.append({**info, "rows": len(frame), "columns": len(frame.columns)})
    if not parsed:
        return None, None, None
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
    columns = st.columns(3)
    learning_rate = float(columns[0].number_input("Learning rate", 1e-5, 2., .01, format="%.5f", key=key + "lr"))
    epochs = int(columns[1].number_input("Эпох за порцию", 1, 5000, 20, key=key + "epochs"))
    seed = int(columns[2].number_input("Seed", 0, 999999, 42, key=key + "seed"))
    with st.expander("Дополнительные параметры"):
        optimizer = st.selectbox("Оптимизатор", ["adam", "sgd"], key=key + "optimizer")
        init_scale = float(st.number_input("Масштаб инициализации", .01, 20., 1., key=key + "scale"))
        tolerance = float(st.number_input("Допуск GF/BF", .0001, 100., .1, key=key + "tol"))
    return layers, learning_rate, epochs, seed, optimizer, init_scale, tolerance


def render_lab07():
    st.title("ЛР7 · Как выбрать модель")
    st.caption("Пять этапов учебного исследования: данные → постановка → обучение → сравнение → Orange.")
    teacher = st.toggle("Режим преподавателя", key="lab07_teacher")
    if teacher: st.info("Ведущий: просите менять только один фактор, выбирать модель по Test и открывать Control в самом конце.")
    task = st.radio("Выберите маршрут", list(TASK_NAMES), format_func=TASK_NAMES.get, horizontal=True, key="lab07_task")
    key = "lab07_" + task
    stage = st.radio("Этап исследования", STEP_NAMES, horizontal=True, key="lab07_step")
    source, frame, splits, meta = _load_data(task, key)
    if frame is None: return
    st.info(f"Сейчас нужно: пройти этап «{stage}». Источник: {meta['source']}; Train/Test/Control: {len(splits['train'])}/{len(splits['test'])}/{len(splits['control']) if splits.get('control') is not None else 0}.")
    if stage == STEP_NAMES[0]:
        st.subheader("1/5 · Получите и исследуйте данные")
        st.plotly_chart(_plot_source(frame, task), use_container_width=True)
        st.dataframe(frame.head(10), use_container_width=True)
        st.caption(f"Строк: {len(frame)} · столбцов: {len(frame.columns)} · доля пропусков: {frame.isna().mean().mean():.1%} · подсказка типа: {TASK_NAMES[suggest_task(frame)]}")
        st.text_area("Что за задача и почему?", key=key + "hypothesis")
        return
    schema = _prepare_schema(task, frame, splits, key, meta["source"])
    if schema is None: return
    features, target, classes, prepared_splits, time_column, value_column, lag = schema
    if stage == STEP_NAMES[1]:
        st.subheader("2/5 · Определите задачу и подготовьте признаки")
        st.write(f"Features → Target: `{features}` → `{target}`")
        st.plotly_chart(_plot_source(frame, task, None if task == "timeseries" else target), use_container_width=True)
        st.success("Схема готова. Перейдите к этапу 3, чтобы запустить реальные обновления весов.")
        return
    if task == "classification":
        classes = list(classes); y_train = one_hot(prepared_splits["train"][target].tolist(), classes); y_test = one_hot(prepared_splits["test"][target].tolist(), classes); output_dim = len(classes)
    else:
        y_train = pd.to_numeric(prepared_splits["train"][target], errors="raise").to_numpy(float).reshape(-1, 1); y_test = pd.to_numeric(prepared_splits["test"][target], errors="raise").to_numpy(float).reshape(-1, 1); output_dim = 1
    x_train_raw, x_test_raw = prepared_splits["train"][features], prepared_splits["test"][features]
    preprocessor = fit_preprocessor(x_train_raw, features)
    x_train, x_test = preprocessor.transform(x_train_raw), preprocessor.transform(x_test_raw)
    layers, learning_rate, epochs, seed, optimizer, init_scale, tolerance = _settings(task, x_train.shape[1], output_dim, key)
    data_fp = experiment_fingerprint({"train": prepared_splits["train"], "test": prepared_splits["test"], "control": prepared_splits.get("control")}, task, features, target, time_column=time_column, value_column=value_column, lag=lag, preprocessing={"features": features})
    model_fp = model_fingerprint(data_fingerprint=data_fp, architecture=layers, optimizer=optimizer, learning_rate=learning_rate, seed=seed, init_scale=init_scale, task=task)
    old_fp = st.session_state.get(key + "fingerprint")
    engine = st.session_state.get(key + "engine")
    if old_fp and old_fp != model_fp:
        st.warning("Изменились данные или параметры модели. Начните новый эксперимент: старые веса не применяются.")
        engine = None
        # Control и lock принадлежат старому fingerprint и не могут отображаться снова.
        st.session_state.pop(key + "control_result", None)
        st.session_state.pop(key + "locked_model", None)
        st.session_state.pop(key + "experiments", None)
    st.caption(f"Fingerprint данных/модели: `{short_fingerprint(model_fp)}`")
    if stage == STEP_NAMES[2]:
        st.subheader("3/5 · Обучите модель")
        st.caption(f"Схема MLP: {x_train.shape[1]} → {' → '.join(map(str, layers[1:-1])) or 'без скрытого слоя'} → {output_dim}")
        actions = st.columns(3)
        if actions[0].button("Запустить обучение", type="primary", key=key + "start"):
            engine = Lab07Engine(layers, task, seed, init_scale=init_scale, optimizer=optimizer, learning_rate=learning_rate, tolerance=tolerance)
            engine.train_epochs(epochs, x_train, y_train, x_test, y_test); st.session_state[key + "engine"] = engine; st.session_state[key + "fingerprint"] = model_fp; st.session_state.pop(key + "control_result", None); st.rerun()
        if actions[1].button(f"Продолжить +{epochs} эпох", key=key + "continue") and engine is not None:
            engine.stopped = False; engine.train_epochs(epochs, x_train, y_train, x_test, y_test); st.session_state[key + "engine"] = engine
        if actions[2].button("Начать новый эксперимент", key=key + "reset"):
            for suffix in ("engine", "fingerprint", "control_result", "locked_model", "experiments"): st.session_state.pop(key + suffix, None)
            st.rerun()
        if engine is None:
            st.info("Пока нет обученной модели. Запустите обучение выше."); return
        prediction = engine.predict(x_test)
        if task == "classification":
            predicted_labels = np.asarray([classes[index] for index in np.argmax(prediction, 1)], dtype=object); metrics = classification_metrics(prepared_splits["test"][target].tolist(), predicted_labels, classes); primary, _ = _primary_plot(task, prepared_splits["test"], target, prediction, classes, features=features, engine=engine, preprocessor=preprocessor)
        elif task == "timeseries":
            baseline = persistence_baseline(prepared_splits["test"], ["lag_1"]); metrics = forecasting_metrics(y_test, prediction, tolerance); primary, _ = _primary_plot(task, prepared_splits["test"], target, prediction, baseline=baseline)
        else:
            metrics = regression_metrics(y_test, prediction, tolerance); primary, _ = _primary_plot(task, prepared_splits["test"], target, prediction)
        st.plotly_chart(primary, use_container_width=True)
        metric_columns = st.columns(4)
        if task == "classification": metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}"); metric_columns[1].metric("Macro-F1", f"{metrics['macro_f1']:.3f}"); metric_columns[2].metric("GF/BF", f"{metrics['gf']}/{metrics['bf']}")
        else: metric_columns[0].metric("Test RMSE", f"{metrics['rmse']:.4f}"); metric_columns[1].metric("Test MAE", f"{metrics['mae']:.4f}"); metric_columns[2].metric("GF/BF", f"{metrics['gf']}/{metrics['bf']}")
        metric_columns[3].metric("Эпоха", engine.epoch)
        history = pd.DataFrame(engine.history)
        if not history.empty: st.plotly_chart(go.Figure([go.Scatter(x=history.epoch, y=history.train_loss, name="Train"), go.Scatter(x=history.epoch, y=history.validation_loss, name="Test")]).update_layout(title="Что заметить: разрыв Train/Test показывает риск переобучения"), use_container_width=True)
        with st.expander("Технические детали"):
            st.write(metrics); st.write("Параметров:", engine.parameter_count)
        return
    if engine is None:
        st.warning("Сначала обучите модель на этапе 3."); return
    prediction = engine.predict(x_test)
    if task == "classification":
        predicted_labels = np.asarray([classes[index] for index in np.argmax(prediction, 1)], dtype=object); metrics = classification_metrics(prepared_splits["test"][target].tolist(), predicted_labels, classes)
    elif task == "timeseries": metrics = forecasting_metrics(y_test, prediction, tolerance)
    else: metrics = regression_metrics(y_test, prediction, tolerance)
    if stage == STEP_NAMES[3]:
        st.subheader("4/5 · Сравните эксперименты и зафиксируйте выбор")
        if st.button("Сохранить эксперимент для сравнения", key=key + "save"):
            st.session_state.setdefault(key + "experiments", []).append({"fingerprint": model_fp, "snapshot": engine.snapshot(), "metrics": metrics, "architecture": "-".join(map(str, layers)), "parameters": engine.parameter_count, "epoch": engine.epoch, "note": st.session_state.get(key + "hypothesis", "")})
        experiments = st.session_state.get(key + "experiments", [])
        if experiments: st.dataframe(pd.DataFrame([{k: item[k] for k in ("fingerprint", "architecture", "parameters", "epoch", "metrics")} for item in experiments]), hide_index=True, use_container_width=True)
        if st.button("Зафиксировать выбранную модель", type="primary", key=key + "lock"):
            st.session_state[key + "locked_model"] = {"fingerprint": model_fp, "epoch": engine.epoch, "snapshot": engine.snapshot()}
            st.session_state.pop(key + "control_result", None)
        if st.session_state.get(key + "locked_model"): st.success("Модель зафиксирована. Теперь доступна финальная оценка Control на следующем этапе.")
        return
    if stage == STEP_NAMES[4]:
        st.subheader("5/5 · Экспорт и Orange")
        locked = st.session_state.get(key + "locked_model")
        if splits.get("control") is not None and not locked: st.warning("Сначала зафиксируйте выбранную модель на этапе 4.")
        control_result = st.session_state.get(key + "control_result")
        if splits.get("control") is not None and locked and locked["fingerprint"] == model_fp and st.button("Итоговая проверка Control", key=key + "control"):
            control = prepared_splits["control"]; xc = preprocessor.transform(control[features]); yc = pd.to_numeric(control[target], errors="raise").to_numpy(float).reshape(-1, 1) if task != "classification" else control[target].tolist()
            if task == "classification": control_result = classification_metrics(yc, [classes[index] for index in np.argmax(engine.predict(xc), 1)], classes)
            else: control_result = forecasting_metrics(yc, engine.predict(xc), tolerance)
            st.session_state[key + "control_result"] = {"fingerprint": model_fp, "epoch": engine.epoch, "metrics": control_result}
            control_result = st.session_state[key + "control_result"]
        if control_result: st.success(f"Control относится к fingerprint {short_fingerprint(control_result['fingerprint'])}, эпохе {control_result['epoch']}."); st.write(control_result["metrics"])
        elif splits.get("control") is None: st.info("Контрольной выборки нет; независимая финальная оценка недоступна.")
        test_predictions = prepared_splits["test"].copy(); test_predictions["prediction"] = predicted_labels if task == "classification" else prediction[:, 0]
        control_predictions = None
        if control_result and prepared_splits.get("control") is not None:
            control_predictions = prepared_splits["control"].copy()
            control_values = engine.predict(preprocessor.transform(prepared_splits["control"][features]))
            control_predictions["prediction"] = ([classes[index] for index in np.argmax(control_values, 1)] if task == "classification" else control_values[:, 0])
        manifest = {"source": meta["source"], "task": task, "features": features, "target": target, "time": time_column, "value": value_column, "lag": lag, "seed": seed, "fingerprint": model_fp, "train_rows": len(prepared_splits["train"]), "test_rows": len(prepared_splits["test"]), "control_rows": len(prepared_splits["control"]) if prepared_splits.get("control") is not None else 0, "preprocessing": "fitted on Train only", "control_revealed": bool(control_result)}
        export_metrics = {"test": metrics};
        if control_result: export_metrics["control"] = control_result["metrics"]
        package = orange_zip(dataset_manifest=manifest, train=prepared_splits["train"], test=prepared_splits["test"], control=prepared_splits.get("control"), predictions=test_predictions, control_predictions=control_predictions, experiment_config={"layers": layers, "optimizer": optimizer, "learning_rate": learning_rate, "seed": seed, "fingerprint": model_fp}, metrics=export_metrics, report="# ЛР7\n\nРезультаты текущего эксперимента.\n")
        st.download_button("Скачать результаты (.zip)", package, "lab07_results.zip", "application/zip", key=key + "download")
        st.markdown("**Orange:** File(train) → Select Columns → Neural Network → Test & Score; File(test) подключите как Test Data; модель передайте в Predictions. Orange обучает собственную модель, а не открывает веса NumPy-MLP.")
