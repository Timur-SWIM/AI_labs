"""Browser-like smoke checks; skipped in the lean local test environment without Streamlit."""
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_app_home_and_lab07_route_render_without_exception():
    root = Path(__file__).parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("ЛР7 · Как выбрать модель").run()
    assert not app.exception
    assert any(r.label == "Выберите маршрут" for r in app.radio)


def test_timeseries_route_can_start_demo_training():
    root = Path(__file__).parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30)
    app.run()
    app.sidebar.radio[0].set_value("ЛР7 · Как выбрать модель").run()
    next(r for r in app.radio if r.label == "Выберите маршрут").set_value("timeseries")
    next(r for r in app.radio if r.label == "Этап исследования").set_value("3. Обучение")
    app.run()
    next(b for b in app.button if b.label == "Запустить обучение").click()
    app.run()
    assert not app.exception
    assert any(metric.label == "Эпоха" for metric in app.metric)
