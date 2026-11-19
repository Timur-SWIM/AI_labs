import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.lab07_datasets import classification_demo, regression_demo, timeseries_demo
from src.lab07_engine import Lab07Engine, one_hot
from src.lab07_exports import orange_zip
from src.lab07_loader import parse_uploaded, suggest_task
from src.lab07_metrics import classification_metrics, regression_metrics
from src.lab07_preprocessing import fit_preprocessor
from src.lab07_timeseries import chronological_lag_split, make_lagged, persistence_baseline


def test_loader_delimiter_decimal_and_limits():
    frame = parse_uploaded("x;y\n1,2;3,4\n".encode(), delimiter=";", decimal=",")
    assert frame.iloc[0].tolist() == [1.2, 3.4]
    with pytest.raises(ValueError):
        parse_uploaded(b"x;x\n1;2\n")


def test_train_only_preprocessing_and_unknown_category():
    train = pd.DataFrame({"num": [1.0, 3.0], "color": ["red", "blue"]})
    test = pd.DataFrame({"num": [100.0, np.nan], "color": ["green", "red"]})
    processor = fit_preprocessor(train, ["num", "color"])
    transformed = processor.transform(test)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] > 10  # Test не влияет на mean/std Train.
    assert transformed[0, 1:].sum() == 0  # неизвестная категория имеет явную all-zero политику.


def test_engine_regression_gradient_and_resume():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 2)); y = (x[:, :1] * .4 - x[:, 1:] * .2)
    first = Lab07Engine((2, 4, 1), seed=8)
    second = Lab07Engine((2, 4, 1), seed=8)
    first.train_epochs(12, x, y); second.train_epochs(5, x, y); second.train_epochs(7, x, y)
    for a, b in zip(first.weights, second.weights): assert np.allclose(a, b)
    grad, _ = first.gradients(x, y); old = first.weights[0][0, 0]; eps = 1e-6
    first.weights[0][0, 0] = old + eps; plus = first.loss(y, first.predict(x))
    first.weights[0][0, 0] = old - eps; minus = first.loss(y, first.predict(x)); first.weights[0][0, 0] = old
    assert np.isclose(grad[0][0, 0], (plus - minus) / (2 * eps), rtol=1e-4, atol=1e-5)


def test_engine_classification_gradient_and_metrics():
    x = np.array([[-1., 0.], [1., 0.], [0., 1.]])
    y = one_hot(["a", "b", "b"], ["a", "b"])
    engine = Lab07Engine((2, 3, 2), task="classification", seed=1)
    grad, _ = engine.gradients(x, y); old = engine.weights[-1][0, 0]; eps = 1e-6
    engine.weights[-1][0, 0] = old + eps; plus = engine.loss(y, engine.predict(x))
    engine.weights[-1][0, 0] = old - eps; minus = engine.loss(y, engine.predict(x)); engine.weights[-1][0, 0] = old
    assert np.isclose(grad[-1][0, 0], (plus - minus) / (2 * eps), rtol=1e-4, atol=1e-5)
    metrics = classification_metrics(["a", "b"], ["a", "a"], ["a", "b"])
    assert metrics["gf"] == 1 and metrics["bf"] == 1


def test_timeseries_is_chronological_and_baseline():
    frame = timeseries_demo(42, 40).frame
    lagged = make_lagged(frame, "value", "time", 3)
    parts = chronological_lag_split(lagged)
    assert parts["train"]["time"].iloc[-1] < parts["test"]["time"].iloc[0]
    assert np.allclose(persistence_baseline(parts["test"], ["lag_1"]), parts["test"]["lag_1"].to_numpy()[:, None])
    with pytest.raises(ValueError): make_lagged(frame.iloc[::-1], "value", "time", 3)


def test_demo_task_suggestions_and_export_hides_control_metrics():
    regression = regression_demo().frame; classification = classification_demo().frame
    assert suggest_task(regression) == "regression"
    assert suggest_task(classification) == "classification"
    train = regression.iloc[:10]; test = regression.iloc[10:15]; control = regression.iloc[15:20]
    payload = orange_zip(dataset_manifest={"task": "regression", "source": "demo", "control_revealed": False}, train=train, test=test, control=control, metrics={"test": {"mse": 1}, "control": {"mse": 2}})
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert "orange_lab07_export/README_ORANGE.md" in names
        assert "orange_lab07_export/control.csv" in names
        assert b"control" not in archive.read("orange_lab07_export/metrics.json")
