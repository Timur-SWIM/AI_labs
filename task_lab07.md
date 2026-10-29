# TASK — ЛР №7 «Прикладные задачи на многослойной нейронной сети»

Источник требований: журнал, страницы 44–48.

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
lab07/
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


## Критическое условие

Искать файлы варианта:
```text
Var_Train.txt
Var_Test.txt
Var_Control.txt
```
Команда поиска:
```bash
find . ~ -type f \( -iname '*_Train.txt' -o -iname '*_Test.txt' -o -iname '*_Control.txt' \) 2>/dev/null
```

Если файлов нет: НЕ придумывать данные варианта. Реализовать универсальный pipeline, создать `BLOCKED_BY_VARIANT_DATA.md` и оставить отчёт без выдуманных экспериментальных результатов. Допускается synthetic smoke-test только для проверки кода.

## Цель

Определить и решить один из типов задач: многомерная аппроксимация, прогноз временного ряда или классификация.

## Теория

Описать постановки трёх задач, подготовку данных, train/test/control, тестирование, генерализацию, overfitting, критерии качества и роль decision block для классификатора.

## Pipeline

1. Загрузчик `src/load_variant.py`: определить delimiter, число входов/target, пропуски, тип данных.
2. Определить тип задачи; при неоднозначности использовать `config.yaml` и записать принятое решение.
3. Сохранить нормализованные `train.csv`, `test.csv`, `control.csv`.
4. Визуализировать данные: distributions, correlations; для классификации PCA-2D, для time series target vs index, для регрессии input/target plots.
5. Обучить базовую MLP.
6. Рассчитать метрики:
   - регрессия: MSE/RMSE/MAE/R²;
   - классификация: Accuracy/Precision/Recall/F1/confusion matrix;
   - прогноз: MAE/RMSE и MAPE только без нулей.
7. Рассчитать GF/BF по порогу из условия. Если порога нет — допущение явно указать.
8. Сравнить минимум 6 архитектур: слои, нейроны, параметры, train/test GF/BF, max/mean error, epochs.
9. Определить underfit/adequate/overfit.
10. Построить улучшенную сеть: уменьшить при overfit, увеличить при underfit.
11. Обработать control и сохранить `control_predictions.csv`.

## Orange

Все три CSV должны открываться Orange.
Рецепт:
```text
File(train) -> Select Columns -> Neural Network
File(test) ---------------------> Predictions
Neural Network -----------------> Predictions
Select Columns(train) ----------> Test and Score
Neural Network -----------------> Test and Score
```
Для классификации добавить Confusion Matrix; для регрессии — Scatter Plot/Python Script.

## Acceptance при наличии файлов
- [ ] Тип задачи определён.
- [ ] Data analysis выполнен.
- [ ] Базовая сеть обучена.
- [ ] Test/control обработаны.
- [ ] GF/BF рассчитаны.
- [ ] Сравнены >=6 архитектур.
- [ ] Улучшенная сеть построена.
- [ ] Orange workflow описан.
- [ ] Отчёт/ответы готовы.

## Acceptance без файлов
- [ ] Pipeline реализован и протестирован synthetic smoke-test.
- [ ] Создан `BLOCKED_BY_VARIANT_DATA.md`.
- [ ] Нет выдуманных значений варианта.
