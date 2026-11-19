"""Streamlit-интерфейс ЛР7; вычисления выполняются в чистых NumPy/pandas модулях."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .lab07_datasets import demo_splits
from .lab07_engine import Lab07Engine, one_hot, validate_layers
from .lab07_exports import orange_zip
from .lab07_loader import parse_file, suggest_task, validate_schema
from .lab07_metrics import classification_metrics, forecasting_metrics, regression_metrics
from .lab07_preprocessing import fit_preprocessor
from .lab07_timeseries import chronological_lag_split, make_lagged, persistence_baseline

_TASK_NAMES = {"regression": "Аппроксимация", "timeseries": "Прогноз временного ряда", "classification": "Классификация"}


def _plot_data(frame, task, target=None):
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    if task == "classification" and len(numeric) >= 2 and target:
        figure = go.Figure()
        for label, part in frame.groupby(target, dropna=False):
            figure.add_scatter(x=part[numeric[0]], y=part[numeric[1]], mode="markers", name=str(label))
        return figure.update_layout(title="Исходные объекты по классам", xaxis_title=numeric[0], yaxis_title=numeric[1])
    if task == "timeseries" and len(frame.columns) >= 2:
        return go.Figure(go.Scatter(x=frame.iloc[:, 0], y=frame.iloc[:, 1], mode="lines+markers", name="ряд")).update_layout(title="Исходный временной ряд")
    if len(numeric) >= 2:
        return go.Figure(go.Scatter(x=frame[numeric[0]], y=frame[numeric[-1]], mode="markers", name="данные")).update_layout(title="Исходные точки")
    return go.Figure()


def _load_source(task, key):
    source = st.radio("Источник", ["demo", "upload"], format_func=lambda value: "Демонстрационный набор" if value == "demo" else "Загрузить файлы варианта", horizontal=True, key=key + "source")
    if source == "demo":
        dataset, splits = demo_splits(task, 42)
        st.warning("Демонстрационный набор — это синтетические данные семинара, не индивидуальный вариант.")
        st.caption(f"Формула: {dataset.formula}; seed={dataset.seed}")
        return source, dataset.frame, splits, {"source": source, "name": dataset.name, "formula": dataset.formula, "seed": dataset.seed}
    uploads = st.file_uploader("Train/Test/Control: .txt, .csv или .tsv; Train обязателен", type=["txt", "csv", "tsv"], accept_multiple_files=True, key=key + "files")
    if not uploads:
        st.info("Выберите файлы через браузер. Серверные пути не используются.")
        return source, None, None, None
    parsed = {}
    info = []
    for uploaded in uploads:
        try:
            table = parse_file(uploaded.name, uploaded.getvalue())
            parsed[uploaded.name] = table.frame
            info.append({"файл": uploaded.name, "строк": len(table.frame), "столбцов": len(table.frame.columns), "байт": table.size_bytes, "SHA-256": table.sha256[:16] + "…", "кодировка": table.encoding, "разделитель": table.delimiter})
        except ValueError as exc:
            st.error(f"{uploaded.name}: {exc}")
    if not parsed:
        return source, None, None, None
    st.dataframe(pd.DataFrame(info), hide_index=True, use_container_width=True)
    names = list(parsed)
    train_name = st.selectbox("Файл Train", names, index=next((i for i, name in enumerate(names) if "train" in name.lower()), 0), key=key + "trainname")
    test_options = ["— отсутствует —"] + [name for name in names if name != train_name]
    test_name = st.selectbox("Файл Test (для выбора модели)", test_options, key=key + "testname")
    control_options = ["— отсутствует —"] + [name for name in names if name != train_name and name != test_name]
    control_name = st.selectbox("Файл Control (финальная оценка)", control_options, key=key + "controlname")
    train = parsed[train_name].copy()
    test = parsed[test_name].copy() if test_name != "— отсутствует —" else None
    control = parsed[control_name].copy() if control_name != "— отсутствует —" else None
    if test is None:
        rng = np.random.default_rng(42)
        order = rng.permutation(len(train)); cut = max(1, int(.8 * len(train)))
        test = train.iloc[order[cut:]].copy().reset_index(drop=True)
        train = train.iloc[order[:cut]].copy().reset_index(drop=True)
        st.warning("Test не загружен: Train разделён воспроизводимо seed=42; Control отсутствует.")
    splits = {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True), "control": None if control is None else control.reset_index(drop=True)}
    return source, pd.concat([train, test, control], ignore_index=True) if control is not None else pd.concat([train, test], ignore_index=True), splits, {"source": source, "name": train_name, "seed": 42}


def _controls(task, input_dim, output_dim, key):
    hidden = st.text_input("Скрытые слои (1–4 слоя, 1–32 нейрона)", "8,8", key=key + "hidden")
    try:
        layers = validate_layers((input_dim, *(int(value.strip()) for value in hidden.split(",") if value.strip()), output_dim), input_dim, output_dim)
    except ValueError as exc:
        st.error(str(exc)); layers = (input_dim, 8, output_dim)
    columns = st.columns(5)
    optimizer = columns[0].selectbox("Оптимизатор", ["adam", "sgd"], key=key + "optimizer")
    learning_rate = columns[1].number_input("Learning rate", 1e-5, 2.0, .01, format="%.5f", key=key + "lr")
    epochs = columns[2].number_input("Эпох за порцию", 1, 5000, 100, key=key + "epochs")
    seed = int(columns[3].number_input("Seed", 0, 999999, 42, key=key + "seed"))
    tolerance = columns[4].number_input("Допуск GF", .0001, 100.0, .1, key=key + "tolerance")
    return layers, optimizer, float(learning_rate), int(epochs), seed, float(tolerance)


def _prepare(task, frame, splits, key):
    if task == "timeseries":
        value_column = st.selectbox("Столбец ряда", frame.columns, key=key + "value")
        time_column = st.selectbox("Столбец времени (необязательно)", ["— индекс —"] + list(frame.columns), key=key + "time")
        lag = st.slider("Lag/window", 1, 64, 3, key=key + "lag")
        source_frame = frame[[value_column] + ([] if time_column == "— индекс —" else [time_column])].copy()
        if time_column != "— индекс —": source_frame = source_frame.sort_values(time_column).reset_index(drop=True)
        lagged = make_lagged(source_frame, value_column, None if time_column == "— индекс —" else time_column, lag)
        splits = chronological_lag_split(lagged)
        features, target, classes = [f"lag_{i}" for i in range(1, lag + 1)], "target", None
    else:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        default_features = numeric[:min(2, len(numeric))]
        features = st.multiselect("Features", frame.columns.tolist(), default=default_features, key=key + "features")
        target_options = [column for column in frame.columns if column not in features]
        if not target_options:
            st.error("Выберите хотя бы один признак так, чтобы отдельный столбец остался Target."); return None
        target = st.selectbox("Target", target_options, key=key + "target")
        validate_schema(splits["train"], splits.get("test"), splits.get("control"), features, target)
        classes = list(dict.fromkeys(splits["train"][target].dropna().tolist())) if task == "classification" else None
        if task == "classification" and len(classes) < 2:
            st.error("В Train должны присутствовать минимум два класса"); return None
    return features, target, classes, splits


def render_lab07():
    st.divider(); st.header("🧪 ЛР7 — прикладные задачи обработки данных")
    teacher = st.toggle("Режим преподавателя", key="lab07_teacher")
    if teacher:
        st.info("Ожидайте: большая сеть обычно уменьшает Train Loss, но не обязана улучшать Test. Control открывается только после фиксации модели.")
    task = st.radio("Шаг 2. Определите тип задачи", list(_TASK_NAMES), format_func=_TASK_NAMES.get, horizontal=True, key="lab07_task")
    key = "lab07_" + task
    source, frame, splits, meta = _load_source(task, key)
    if frame is None: return
    st.caption(f"Набор: {meta['name']} · источник: {meta['source']} · Train/Test/Control: {len(splits['train'])}/{len(splits['test'])}/{len(splits['control']) if splits.get('control') is not None else 0} · seed={meta['seed']}")
    st.subheader("Шаг 1. Данные")
    st.plotly_chart(_plot_data(frame, task), use_container_width=True); st.dataframe(frame.head(10), use_container_width=True)
    st.caption("Автоподсказка типа задачи не является доказательством: " + _TASK_NAMES[suggest_task(frame)])
    prepared = _prepare(task, frame, splits, key)
    if prepared is None: return
    features, target, classes, splits = prepared
    st.subheader("Шаг 2. Постановка задачи")
    st.write("Features:", features, "Target:", target)
    notes = st.text_area("Гипотеза → Изменение → Наблюдение → Вывод", key=key + "notes")
    if task == "timeseries":
        x_train_raw = splits["train"][features]; x_test_raw = splits["test"][features]; y_train_raw = splits["train"][target].to_numpy(float).reshape(-1, 1); y_test_raw = splits["test"][target].to_numpy(float).reshape(-1, 1); output_dim = 1
    elif task == "classification":
        x_train_raw = splits["train"][features]; x_test_raw = splits["test"][features]; y_train_raw = one_hot(splits["train"][target].tolist(), classes); y_test_raw = one_hot(splits["test"][target].tolist(), classes); output_dim = len(classes)
    else:
        x_train_raw = splits["train"][features]; x_test_raw = splits["test"][features]; y_train_raw = pd.to_numeric(splits["train"][target], errors="raise").to_numpy(float).reshape(-1, 1); y_test_raw = pd.to_numeric(splits["test"][target], errors="raise").to_numpy(float).reshape(-1, 1); output_dim = 1
    preprocessor = fit_preprocessor(x_train_raw, features)
    x_train, x_test = preprocessor.transform(x_train_raw), preprocessor.transform(x_test_raw)
    layers, optimizer, learning_rate, epochs, seed, tolerance = _controls(task, x_train.shape[1], output_dim, key)
    signature = (layers, task, seed, optimizer, learning_rate, tuple(features), target)
    engine = st.session_state.get(key + "engine")
    if st.session_state.get(key + "signature") != signature:
        if engine is not None: st.warning("Изменились данные/архитектура: нажмите Reset для нового эксперимента.")
        engine = None
    st.subheader("Шаг 3. Обучение")
    buttons = st.columns(4)
    if buttons[0].button("Reset", key=key + "reset"):
        st.session_state.pop(key + "engine", None); st.session_state.pop(key + "signature", None); st.session_state.pop(key + "experiments", None); st.rerun()
    if buttons[1].button("Обучить / Continue", key=key + "train"):
        if engine is None:
            engine = Lab07Engine(layers, task, seed, optimizer=optimizer, learning_rate=learning_rate, tolerance=tolerance)
            st.session_state[key + "signature"] = signature
        engine.stopped = False; engine.train_epochs(epochs, x_train, y_train_raw, x_test, y_test_raw); st.session_state[key + "engine"] = engine
    if buttons[2].button("Остановить", key=key + "stop") and engine is not None: engine.stopped = True
    if buttons[3].button("Сохранить эксперимент", key=key + "save") and engine is not None:
        st.session_state.setdefault(key + "experiments", []).append({"architecture": "-".join(map(str, engine.layers)), "parameters": engine.parameter_count, "epochs": engine.epoch, "test_loss": engine.history[-1]["validation_loss"] if engine.history else None, "note": notes})
    if engine is None:
        st.info("Зафиксируйте схему и запустите обучение."); return
    history = pd.DataFrame(engine.history)
    if not history.empty: st.plotly_chart(go.Figure([go.Scatter(x=history.epoch, y=history.train_loss, name="Train"), go.Scatter(x=history.epoch, y=history.validation_loss, name="Test")]).update_layout(title="Реальные кривые Loss"), use_container_width=True)
    prediction = engine.predict(x_test)
    if task == "classification":
        predicted_labels = np.asarray([classes[index] for index in np.argmax(prediction, axis=1)], dtype=object); metrics = classification_metrics(splits["test"][target].tolist(), predicted_labels, classes); st.json({key_: value for key_, value in metrics.items() if key_ != "confusion_matrix"}); st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)); st.plotly_chart(go.Figure(go.Scatter(x=x_test[:, 0], y=x_test[:, 1] if x_test.shape[1] > 1 else np.zeros(len(x_test)), mode="markers", marker_color=np.argmax(prediction, axis=1), name="decision points")).update_layout(title="Точки и текущие предсказанные классы"), use_container_width=True)
    else:
        metrics = forecasting_metrics(y_test_raw, prediction, tolerance); st.json(metrics); st.plotly_chart(go.Figure([go.Scatter(y=y_test_raw[:, 0], mode="lines+markers", name="true"), go.Scatter(y=prediction[:, 0], mode="lines+markers", name="predicted")]).update_layout(title="Предметное предсказание"), use_container_width=True); st.plotly_chart(go.Figure(go.Scatter(y=(y_test_raw - prediction)[:, 0], mode="lines+markers", name="residual")).update_layout(title="Остатки"), use_container_width=True)
    st.subheader("Шаг 4. Сравнение моделей")
    st.metric("Параметров", engine.parameter_count); st.caption("Выбор модели делается по Test; Control не участвует в подборе.")
    experiments = st.session_state.get(key + "experiments", [])
    if experiments: st.dataframe(pd.DataFrame(experiments), hide_index=True, use_container_width=True)
    control_result = None
    if splits.get("control") is not None and st.button("Открыть финальную оценку Control", key=key + "reveal"):
        control = splits["control"]
        if task == "timeseries":
            xc = preprocessor.transform(control[features]); yc = control[target].to_numpy(float).reshape(-1, 1); control_result = forecasting_metrics(yc, engine.predict(xc), tolerance)
        elif task == "classification":
            xc = preprocessor.transform(control[features]); yc = control[target].tolist(); control_result = classification_metrics(yc, [classes[i] for i in np.argmax(engine.predict(xc), 1)], classes)
        else:
            xc = preprocessor.transform(control[features]); yc = pd.to_numeric(control[target], errors="raise").to_numpy(float).reshape(-1, 1); control_result = regression_metrics(yc, engine.predict(xc), tolerance)
        st.session_state[key + "control_result"] = control_result
    if st.session_state.get(key + "control_result") is not None: st.warning("Control раскрыт после выбора модели; ретроспективная настройка по нему нарушает независимость."); st.json(st.session_state[key + "control_result"])
    st.subheader("Шаг 5. Orange и экспорт")
    test_predictions = splits["test"].copy(); test_predictions["prediction"] = predicted_labels if task == "classification" else prediction[:, 0]
    manifest = {**meta, "task": task, "features": features, "target": target, "classes": classes, "train_rows": len(splits["train"]), "test_rows": len(splits["test"]), "control_rows": len(splits["control"]) if splits.get("control") is not None else 0, "preprocessing": "imputation/one-hot and standardization fitted on Train only", "control_revealed": st.session_state.get(key + "control_result") is not None}
    metrics_export = {"test": metrics};
    if st.session_state.get(key + "control_result") is not None: metrics_export["control"] = st.session_state[key + "control_result"]
    package = orange_zip(dataset_manifest=manifest, train=splits["train"], test=splits["test"], control=splits.get("control"), predictions=test_predictions, experiment_config={"layers": layers, "optimizer": optimizer, "learning_rate": learning_rate, "seed": seed}, metrics=metrics_export, report="# ЛР7\n\nРезультаты вычислены текущим запуском.\n\n" + notes)
    st.download_button("Скачать пакет для Orange (.zip)", package, "orange_lab07_export.zip", "application/zip", key=key + "download")
    st.download_button("Скачать заметки Markdown", (f"# ЛР7 — {_TASK_NAMES[task]}\n\n{notes}\n").encode("utf-8"), "lab07_notes.md", "text/markdown", key=key + "notesdownload")
    st.info("Самопроверка: почему низкий Train Loss не гарантирует качество на новых данных?")
