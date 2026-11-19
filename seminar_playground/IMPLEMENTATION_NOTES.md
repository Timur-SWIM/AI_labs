# Заметки реализации

ЛР6 использует прежний `training_engine.py`; вычислительные формулы не менялись. Навигация теперь выполняется в начале `app.py`, поэтому ЛР7 не находится после длинного экрана ЛР6.

ЛР7 использует `Lab07Engine`: произвольные входы/выходы, tanh, linear/MSE для regression и forecasting, softmax/cross-entropy для classification. `timeseries` разрешён как семантический alias регрессионного выхода.

`lab07_state.py` считает SHA-256 fingerprint канонических Train/Test/Control таблиц, схемы, preprocessing и модели. Изменение fingerprint удаляет старые engine/lock/control state. GF/BF и метрики Control считаются только для зафиксированного snapshot.

`make_lagged_parts` строит временные окна по общей хронологии, но оставляет target в исходном split; первые Test/Control окна используют только прошлый контекст.

Orange GUI в окружении разработки не установлен: ZIP и инструкция проверяются как файлы, `.ows` не создаются.
