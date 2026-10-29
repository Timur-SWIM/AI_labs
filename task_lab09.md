# TASK — ЛР №9 «Самообучение нейронных сетей. Кластеризация данных»

Источник требований: журнал, страницы 54–58.

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
lab09/
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

Реализовать competitive learning и SOM на NumPy; исследовать кластеризацию, негативные эффекты и карты Кохонена.

## Теория

Описать кластеризацию, unsupervised learning, prototype/centroid, winner-takes-all, Euclidean distance, learning rate, epoch, dead neuron, merging, splitting/detailing, oscillation, SOM, neighborhood, 1D/2D map. Привести формулы BMU и обновления весов.

## Эксперимент 1: 7 кластеров

Согласно журналу: 7 эллипсов, 10 точек в каждом, дисперсия 0.1. Если центры варианта не заданы — создать 7 воспроизводимых хорошо разделимых центров и записать их в `config.yaml`.

Обучить competitive layer с 7 нейронами. Сохранить initial/final centers, epochs, learning rate, quantization error. Графики initial/final/error.

## Эксперимент 2: негативные эффекты

Согласно журналу: 5 эллипсов, 10 точек, дисперсия 0.2.

Автоматически подобрать и продемонстрировать:
1. укрупнение кластеров;
2. детализацию;
3. мёртвые нейроны;
4. колебательный режим.

Объективные критерии:
- dead neuron: долго ни разу не BMU;
- merging: эффективных прототипов меньше естественных групп;
- detailing: несколько устойчивых прототипов внутри одной группы;
- oscillation: центры/ошибка не стабилизируются.

Сохранить таблицу параметров и PNG каждого эффекта.

## Эксперимент 3: контрольная выборка

Создать 7 контрольных точек у известных кластеров. Сохранить x,y,predicted_cluster; показать на графике training points + centers + control.

## Эксперимент 4: одномерные данные

Сгенерировать 1D выборку с неравномерной плотностью. Обучить competitive layer. Построить density, final weights, weights vs epoch.

## Эксперимент 5: 2D SOM на 2D данных

Реализовать rectangular SOM: BMU, neighborhood, убывающий radius и learning rate. Показать initial map, final embedding, U-matrix/neighbor distances.

## Эксперимент 6: 1D SOM на 2D данных

Обучить цепочку нейронов на 2D данных. Показать порядок нейронов на карте и их положение в feature space; оценить топологическое соответствие.

## Hyperparameter sweep

Перебрать число нейронов, initial LR, epochs, neighborhood radius. Рассчитать quantization error, число dead neurons, число занятых кластеров. Сохранить CSV и summary plot.

## Orange

Core Orange использовать для просмотра данных/кластеров, модель остаётся Python:
```text
File -> Scatter Plot
File -> Python Script -> Data Table
                      -> Scatter Plot
```
Подготовить CSV: input points, centers, control points, SOM weights, epoch/error. Не требовать стороннего SOM add-on.

## Acceptance
- [ ] Competitive learning реализован самостоятельно.
- [ ] 7-cluster experiment выполнен.
- [ ] Показаны 4 негативных эффекта.
- [ ] Control points обработаны.
- [ ] 1D data исследованы.
- [ ] 2D SOM реализована.
- [ ] 1D SOM на 2D реализована.
- [ ] Hyperparameter sweep выполнен.
- [ ] CSV/PNG готовы.
- [ ] Orange открывает результаты.
- [ ] Отчёт и ответы готовы.
