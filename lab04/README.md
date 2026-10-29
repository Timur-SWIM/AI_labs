# ЛР №4 — примеры использования NNTool

Headless-реализация на Python/scikit-learn с CSV для Orange.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
pytest -q
```

`run_all.py` пересоздаёт `data/`, `results/` и `plots/`. Результаты не
зашиваются в исходный код: все числа в отчёте должны быть получены запуском.
Описание подключения виджетов Orange находится в `orange_workflow.md`.

