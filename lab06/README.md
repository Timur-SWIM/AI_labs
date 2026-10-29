# ЛР №6 — MLP и backpropagation

Лабораторная исследует преобразование полярных координат в декартовы:
`(ρ, φ) → (x, y)`, `x=ρcosφ`, `y=ρsinφ`. Основной эксперимент использует
собственный прозрачный NumPy-MLP с архитектурой 2–7–3–2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
pytest -q
```

Команда создаёт 100 примеров, train/test/control-разбиение, выполняет sweep
learning rate, плохие режимы, остановку при 80% GF и сравнение пяти
архитектур. Все результаты записываются в `results/`, графики — в `plots/`.

