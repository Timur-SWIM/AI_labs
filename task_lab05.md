# TASK — ЛР №5 «Сеть Хемминга»

Источник требований: журнал, страницы 32–38.

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
lab05/
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

Самостоятельно реализовать классификатор `образ -> слой Хемминга -> MAXNET -> решение`, исследовать feed-forward и recurrent MAXNET, шум и параметр ε.

## Теория

Описать технический нейрон, синаптические коэффициенты, feed-forward/recurrent сети, расстояние Хемминга, сеть Хемминга, потенциалы, MAXNET, компаратор, ε и критерий победителя. Привести формулы используемой модели.

## Данные

Сначала искать изображения/матрицы варианта. Если их нет — создать 3 демонстрационных бинарных образа одинакового размера (5×5 или 7×7) и явно отметить отсутствие варианта. Сохранить CSV и `plots/reference_patterns.png`.

## Реализация

Создать:
- `src/hamming.py` — биполяризация, сходство, потенциалы;
- `src/maxnet_feedforward.py` — прозрачный выбор максимума через компараторы, не только `argmax`;
- `src/maxnet_recurrent.py` — взаимное торможение, итерации до одного победителя.

## Эксперименты

1. Точные эталоны: потенциалы, победитель обеих MAXNET, число шагов.
2. Искажение входа: для каждого Hamming distance выполнять серию случайных искажений; построить accuracy vs Hamming distance.
3. Возмущение весов: 0%, 1%, 5%, 10%, 20%; построить accuracy vs perturbation.
4. Sweep ε: минимум 8 значений; сохранить число итераций и корректность; построить `iterations_vs_epsilon.png`.

Сохранить CSV всех экспериментов.

## Orange

Основная модель через Python. Orange workflow:
```text
File -> Python Script -> Data Table
                      -> Line Plot / Scatter Plot
```
CSV должны содержать динамику, шум/accuracy, ε/iterations и итоговые результаты.

## Отчёт

Включить три класса, схему Input→Hamming→MAXNET→Decision, модель нейрона, логику компаратора, обе MAXNET, устойчивость к шуму, влияние весов и ε.

## Acceptance
- [ ] Hamming реализован самостоятельно.
- [ ] Две MAXNET реализованы.
- [ ] Исследованы 3 класса.
- [ ] Выполнены шум, веса, ε.
- [ ] Есть CSV/PNG.
- [ ] Orange может открыть результаты.
- [ ] Отчёт и ответы готовы.
