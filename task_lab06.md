# TASK — ЛР №6 «Математические модели искусственных нейронных сетей»

Источник требований: журнал, страницы 39–43.

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
lab06/
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

Исследовать MLP и backpropagation на преобразовании `(ρ,φ)->(x,y)`, где `x=ρ cosφ`, `y=ρ sinφ`.

## Теория

Описать MLP, прямой проход, loss, backpropagation, learning rate, эпоху, train/test/control, генерализацию, overfitting, «паралич»/насыщение, влияние архитектуры и скорости обучения.

## Данные

Если `coord.dat` найден — использовать. Иначе создать 100 воспроизводимых примеров первого квадранта: `ρ∈[0.05,1]`, `φ∈[0,π/2]`. Сохранить `coord.csv` и train/test/control split.

## Базовая сеть

Основная архитектура 2–7–3–2 (10–12 нейронов согласно журналу). Реализация NumPy или прозрачная MLP-модель с доступом к весам и кривой обучения.

## Эксперименты

1. Базовое обучение: train/validation loss по эпохам.
2. Learning-rate sweep: 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1.
3. Плохие режимы: отдельно воспроизвести насыщение/паралич и слишком высокий learning rate.
4. Остановить одну тренировку, когда доля примеров в заданной точности впервые достигает ~80%; затем протестировать.
5. Архитектуры: 4–4–2, 6–4–2, 7–3–2, 8–4–2, 10–6–2. Для каждой: число параметров, GF/BF, train/test error, epochs.
6. Выбрать улучшенную сеть по test error/генерализации, а не только train loss.

Графики: loss curves, learning-rate comparison, paralysis case, high-LR case, x true/pred, y true/pred, error distribution.

GF = примеры в допуске, BF = вне допуска. Порог брать из условия/конфига, иначе явно отметить допущение.

## Orange

Для совместимости подготовить два набора: `(rho,phi)->x` и `(rho,phi)->y`.
Workflow:
```text
File -> Select Columns -> Neural Network
Select Columns -> Predictions
Neural Network -> Predictions
Select Columns -> Test and Score
Neural Network -> Test and Score
```
Основной двухвыходный эксперимент остаётся Python.

## Acceptance
- [ ] 100 примеров готовы.
- [ ] 2–7–3–2 исследована.
- [ ] Есть настоящие loss curves.
- [ ] Выполнен LR sweep.
- [ ] Показаны плохие режимы.
- [ ] Есть остановка около 80%.
- [ ] Сравнены минимум 5 архитектур.
- [ ] Выбрана улучшенная сеть.
- [ ] Orange-compatible CSV готовы.
- [ ] Отчёт и ответы готовы.
