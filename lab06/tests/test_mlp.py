import numpy as np

from src.mlp import MLPRegressor


def test_forward_shape_and_parameter_count():
    model = MLPRegressor([2, 7, 3, 2], seed=42)
    assert model.predict(np.zeros((5, 2))).shape == (5, 2)
    assert model.parameter_count == 53


def test_backprop_reduces_loss():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 2))
    y = np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))
    model = MLPRegressor([2, 6, 2], seed=42)
    initial = model.mse(y, model.predict(x))
    model.fit(x, y, learning_rate=.01, max_epochs=300, optimizer="adam")
    assert model.mse(y, model.predict(x)) < initial * .1


def test_stop_at_target_fraction():
    x = np.zeros((10, 2)); y = np.zeros((10, 2))
    model = MLPRegressor([2, 3, 2], seed=42)
    model.fit(x, y, max_epochs=20, stop_good_fraction=.8, tolerance=.05)
    assert len(model.history) == 1

