# TASK — ЛР №4 «Примеры использования NNTool»

Источник требований: журнал, страницы 25–31.

## Окружение на Ubuntu/Debian

Основная реализация должна быть headless/CLI. Orange используется как совместимая визуальная оболочка, но не как обязательный runtime.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git graphviz xvfb libgl1 libegl1 libxkbcommon-x11-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install numpy pandas scipy matplotlib scikit-learn jupyterlab pillow pyyaml rich pytest
```

Для Orange:

```bash
pip install PyQt6 PyQt6-WebEngine
pip install orange3
python - <<'PY2'
import Orange
print(Orange.__version__)
PY2
```

GUI Orange: `python -m Orange.canvas`. На headless-сервере допускается проверка через `xvfb-run -a python -m Orange.canvas`, но автоматические эксперименты не должны зависеть от GUI.

Для каждой ЛР обязательно подготовить Orange-совместимые CSV, `orange_workflow.md` с точной схемой виджетов и настройками, а также headless Python-реализацию.

## Общие требования Codex

Создать структуру:

```text
lab04/
├── README.md
├── requirements.txt
├── config.yaml
├── run_all.py
├── src/
├── tests/
├── data/
├── results/
├── plots/
├── report/
│   ├── theory.md
│   ├── report.md
│   └── answers.md
└── orange_workflow.md
```

`run_all.py` должен одной командой выполнять все доступные эксперименты и пересоздавать результаты:

```bash
source .venv/bin/activate
python run_all.py
pytest -q
```

Правила:
- фиксировать `random_seed=42`, кроме статистических прогонов;
- все графики сохранять PNG, `dpi >= 150`;
- все численные результаты сохранять CSV/JSON;
- не придумывать результаты вручную;
- если данных варианта нет — явно отметить это, а не подменять их своими;
- `report/report.md` заполнять фактическими значениями только после запуска кода;
- `report/answers.md` должен содержать ответы на все контрольные вопросы журнала для данной ЛР.


## Цель

Современная замена NNTools: Python/scikit-learn + Orange. Выполнить 4 части: логическая функция, аппроксимация, классификация, демонстрационный пример.

## Теория

В `report/theory.md` раскрыть: искусственный нейрон, `z=Σw_ix_i+b`, веса и bias, Identity/Sigmoid/tanh/ReLU, feed-forward MLP, обучение с учителем, loss, backpropagation, SGD/Adam/L-BFGS, эпоха и итерация, train/test/CV, classification vs regression, метрики CA/Precision/Recall/F1/MSE/RMSE/MAE/R².

## 1. XOR

Создать `data/xor.csv`:

```csv
x1,x2,result
0,0,false
0,1,true
1,0,true
1,1,false
```

Обучить `MLPClassifier`:
- hidden `(4,)`;
- activation `logistic`;
- основной solver `adam`, сравнить с `lbfgs`;
- max_iter 5000;
- random_state 42.

Сохранить:
- `results/xor_predictions.csv`;
- `results/xor_metrics.json`;
- `plots/xor_training_curve.png` — реальный `loss_curve_` Adam;
- параметры и фактическое число итераций.

Проверить, что хотя бы одна корректная конфигурация достигает accuracy=1.0. Если нет — исследовать причину, не подменять результат.

## 2. Аппроксимация

Использовать `y=x^2`, 21 равномерная точка `x∈[-3,3]`.

`MLPRegressor`: `(10,)`, ReLU, Adam, max_iter 5000. Выполнить 5-fold CV и сравнить скрытые слои 5 и 10 нейронов.

Рассчитать MSE, RMSE, MAE, R². MAPE не использовать как главный критерий из-за `y=0`.

Сохранить:
- `results/approx_predictions.csv`;
- `results/approx_metrics.csv`;
- `results/approx_weights.json`;
- `plots/approx_true_vs_pred.png`;
- `plots/approx_training_curve.png`;
- `plots/approx_residuals.png`.

## 3. Классификация

Создать `data/classification.csv` с 2 признаками и 2 классами A/B, минимум 10 объектов каждого класса.

MLP: `(4,)`, ReLU, Adam. 5-fold stratified CV. Рассчитать Accuracy, Precision, Recall, F1, confusion matrix.

Графики:
- `classification_data.png`;
- `classification_predictions.png`;
- `classification_decision_regions.png`;
- `classification_confusion_matrix.png`.

## 4. Демонстрационный пример

Если вариант преподавателя не найден — использовать Iris. Сохранить `data/iris.csv`, исследовать 4 признака, 3 класса, MLP `(8,)`, ReLU, Adam, 5-fold CV. Построить scatter и confusion matrix.

## Orange

Подготовить точные схемы:

XOR:
```text
File -> Select Columns -> Neural Network
Select Columns -> Predictions
Neural Network -> Predictions
Select Columns -> Test and Score
Neural Network -> Test and Score
Predictions -> Data Table
```
Для XOR использовать `Test on train data`.

Регрессия/классификация/Iris — те же связи, плюс `Test and Score -> Confusion Matrix` для классификации и `Predictions -> Python Script` для специальных графиков.

Явно предупредить в `orange_workflow.md`:
- Predictions должен получать и Data, и Model;
- Test and Score должен получать и Data, и Learner;
- не использовать Selected Predictions вместо полного Predictions без необходимости.

## Отчёт

`report/report.md`: цель, теория, данные, архитектуры, настройки, таблицы, фактические метрики, кривые обучения, графики, веса/bias для аппроксимации, выводы по каждой части и общий вывод.

## Acceptance
- [ ] 4 части выполнены headless.
- [ ] Есть реальная loss curve XOR.
- [ ] Есть true/pred аппроксимации.
- [ ] Есть decision regions и confusion matrix.
- [ ] Iris/вариант выполнен.
- [ ] CSV открываются Orange.
- [ ] Отчёт и ответы готовы.
