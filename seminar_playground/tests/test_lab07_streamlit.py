from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


def _app(task='regression'):
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('ЛР7 · Как выбрать модель').run()
    next(r for r in app.radio if r.label == 'Выберите маршрут').set_value(task).run()
    return app


def _click(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def test_two_models_survive_navigation():
    app = _app()
    assert not app.get('download_button')
    _click(app, 'Зафиксировать постановку')
    _click(app, 'Далее: обучить две модели')
    assert not app.get('download_button')
    _click(app, 'Запустить обучение')
    _click(app, 'Сохранить эксперимент')
    records = app.session_state['lab07_regressionstudies'][app.session_state['lab07_regressiondata_fingerprint']]
    before = records[0]['test_metrics'].copy()
    _click(app, 'Новый эксперимент')
    next(s for s in app.selectbox if s.label == 'Архитектура').set_value('ручная').run()
    next(s for s in app.text_input if s.label == 'Скрытые слои через запятую').set_value('16').run()
    _click(app, 'Запустить обучение')
    _click(app, 'Сохранить эксперимент')
    assert len(records) == 2 and records[0]['test_metrics'] == before
    assert not app.get('download_button')
    assert not any(b.label == 'Сравнить выбранные' for b in app.button)
    assert any('Test RMSE' in frame.value.columns for frame in app.dataframe)
    _click(app, 'Далее: выбрать модель для отчёта')
    assert any(s.value.startswith('3/3') for s in app.subheader)
    assert not any(s.value.startswith('2/3') for s in app.subheader)
    assert not any(b.label == 'Итоговая проверка Control' for b in app.button)
    assert not app.get('download_button')
    _click(app, 'Зафиксировать выбор')
    assert app.session_state['lab07_regressionlocked_id'] == records[0]['id']
    _click(app, 'Назад')
    _click(app, 'Новый эксперимент')
    _click(app, 'Запустить обучение')
    _click(app, 'Далее: выбрать модель для отчёта')
    _click(app, 'Итоговая проверка Control')
    assert app.session_state['lab07_regressioncontrol_result']['id'] == records[0]['id']
    assert [b.label for b in app.get('download_button')] == ['Скачать результаты ZIP']
    assert len(records) == 2


def test_timeseries_can_train():
    app = _app('timeseries')
    _click(app, 'Зафиксировать постановку')
    _click(app, 'Далее: обучить две модели')
    _click(app, 'Запустить обучение')


def test_classification_can_train():
    app = _app('classification')
    _click(app, 'Зафиксировать постановку')
    _click(app, 'Далее: обучить две модели')
    _click(app, 'Запустить обучение')
