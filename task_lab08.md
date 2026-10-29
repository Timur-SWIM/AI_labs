# TASK — ЛР №8 «Ассоциативная память. Сеть Хопфилда»

Источник требований: журнал, страницы 49–53.

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
lab08/
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

Реализовать Hopfield на NumPy и исследовать динамику, аттракторы, бассейны, ассоциативную память, робастность и ложную память для 1, 3 и 7 образов.

## Теория

Описать автоассоциативную память, аттрактор, правило Хебба, W, симметрию, нулевую диагональ, async/sync update, energy, basin, positive/negative pattern, false memory, Hamming distance, capacity, robustness. Привести уравнения динамики и энергии.

## Данные

Искать вариант. Если нет — создать 7 бинарных образов одинакового размера, достаточно различимых; сохранить CSV и reference plot. Использовать биполярное представление (-1, 1).

## Реализация `src/hopfield.py`

- обучение по Хеббу;
- симметричная W, нулевая диагональ;
- async update (основной);
- sync update для сравнения;
- energy;
- детект устойчивого состояния/цикла;
- Hamming distance.

## 1 образ

Для positive, negative, random start сохранить траекторию, число тактов, начальную/конечную energy, attractor. Построить energy vs iteration.

Бассейны: полный перебор 2^N если реалистично, иначе >=5000 случайных стартов. Оценить positive/negative/false memory fractions.

Ассоциативная память: найти максимальный Hamming noise, при котором образ восстанавливается.

Робастность: сравнить W при вариантах noise+symmetry+zero diag, noise+zero diag без symmetry, noise+symmetry с ненулевой диагональю.

## 3 образа

Повторить динамику, бассейны, recall, robustness. Дополнительно вычислить max/min корреляцию между образами и false-memory fraction.

## 7 образов

Выполнить бассейны, ложную память, robustness, correlations. Сравнить 1/3/7.

## Графики

- energy vs iteration;
- recall accuracy vs Hamming noise;
- recall accuracy vs weight noise;
- basin distributions;
- false memory fraction vs number of stored patterns;
- memory load comparison.

## Orange

Подготовить CSV для Line Plot/Bar Plot/Scatter Plot. Для интерактивного запуска модели:
```text
File -> Python Script -> Data Table
```
В `orange_workflow.md` дать Python Script для одного recall-эксперимента.

## Acceptance
- [ ] Hopfield реализован самостоятельно.
- [ ] 1/3/7 образов исследованы.
- [ ] Energy и attractors рассчитаны.
- [ ] Basins исследованы.
- [ ] Recall/noise исследован.
- [ ] Robustness и false memory исследованы.
- [ ] CSV/PNG готовы.
- [ ] Orange открывает результаты.
- [ ] Отчёт и ответы готовы.
