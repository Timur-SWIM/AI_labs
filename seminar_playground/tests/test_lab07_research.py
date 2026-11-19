import numpy as np
import pytest
from src.lab07_engine import Lab07Engine
from src.lab07_research import save_experiment, restore_engine, export_research, import_research


def test_snapshot_is_independent_and_roundtrips():
    engine = Lab07Engine((2, 8, 1))
    x = np.array([[0., 1.], [1., 2.]])
    y = np.array([[1.], [2.]])
    engine.train_epochs(2, x, y)
    records = []
    save_experiment(records, data_fingerprint='data', model_fingerprint='model', engine=engine,
                    train_metrics={'rmse': 1.}, test_metrics={'rmse': 2.}, name='A', predictions=engine.predict(x))
    before = restore_engine(records[0]['snapshot']).predict(x).copy()
    engine.train_epochs(2, x, y)
    assert np.array_equal(restore_engine(records[0]['snapshot']).predict(x), before)
    imported = import_research(export_research('data', records), 'data')
    assert np.array_equal(restore_engine(imported[0]['snapshot']).predict(x), before)
    with pytest.raises(ValueError): import_research(export_research('data', records), 'other')
