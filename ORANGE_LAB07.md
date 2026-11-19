# Orange для ЛР7

ZIP содержит `train.csv`, `test.csv`, при наличии `control.csv`, `predictions_test.csv`, manifest и конфигурацию. В Orange: `File(train) → Select Columns → Neural Network → Test & Score (Data)`, `File(test) → Test & Score (Test Data)`, а также `Neural Network (Model) → Predictions (Model)` и `File(test) → Predictions (Data)`.

Для классификации проверьте Discrete/Categorical Target и подключите Confusion Matrix. Для регрессии — Continuous Target и Scatter Plot. Для временного ряда используйте готовые `lag_1..lag_N` и `target`: это честный one-step pipeline без автоматического использования будущих данных.

Файлы `.ows` не включаются: Orange GUI в окружении разработки не установлен и не проверен. Пути File после распаковки выберите заново.
