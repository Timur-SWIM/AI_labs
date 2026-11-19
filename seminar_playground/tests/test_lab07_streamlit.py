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
    next(r for r in app.radio if r.label == "Выберите маршрут").set_value("timeseries").run()
    for _ in range(2):
        next(b for b in app.button if b.label == "Далее").click().run()
    next(b for b in app.button if b.label == "Запустить обучение").click()
    app.run()
    assert not app.exception
    assert any(metric.label == "Эпоха" for metric in app.metric)


def test_lab07_only_active_screen_survives_navigation():
    root = Path(__file__).parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("ЛР7 · Как выбрать модель").run()

    def check(step):
        assert not app.exception
        headings = [item.value for item in app.subheader]
        assert len([heading for heading in headings if any(heading.startswith(f"{number}/5") for number in range(1, 6))]) == 1
        assert any(heading.startswith(f"{step}/5") for heading in headings)
        assert any(item.value == step * 20 for item in app.get("progress"))
        labels = {item.label for kind in ("button", "download_button", "radio", "selectbox", "multiselect") for item in app.get(kind)}
        assert ("Источник" in labels) == (step == 1)
        assert ("Features — признаки объекта" in labels) == (step == 2)
        assert ("Запустить обучение" in labels) == (step == 3)
        assert ("Зафиксировать выбранную модель" in labels) == (step == 4)
        assert ("Скачать результаты (.zip)" in labels) == (step == 5)

    check(1)
    for step in range(2, 6):
        next(button for button in app.button if button.label == "Далее").click().run()
        check(step)
        if step == 3:
            next(button for button in app.button if button.label == "Запустить обучение").click().run()
            check(3)
            assert any(metric.label == "Эпоха" for metric in app.metric)
    assert not any(button.label == "Итоговая проверка Control" for button in app.button)
    next(button for button in app.button if button.label == "Назад").click().run()
    check(4)
    next(button for button in app.button if button.label == "Зафиксировать выбранную модель").click().run()
    next(button for button in app.button if button.label == "Далее").click().run()
    check(5)
    assert any(button.label == "Итоговая проверка Control" for button in app.button)
    for step in (4, 3, 2):
        next(button for button in app.button if button.label == "Назад").click().run()
        check(step)
    assert not any(button.label == "Скачать результаты (.zip)" for button in app.button)
