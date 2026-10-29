# ЛР №5 — сеть Хемминга и MAXNET

Headless-реализация классификатора «образ → Hamming layer → MAXNET →
решение». Данные варианта преподавателя в рабочем каталоге отсутствуют,
поэтому используются три демонстрационных образа 5×5: `cross`, `square`,
`diamond`. Это явно отражено в `results/metadata.json` и отчёте.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
pytest -q
```

Результаты сохраняются в `results/`, изображения — в `plots/`. Схема для
Orange описана в `orange_workflow.md`.

