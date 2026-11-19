# ЛР7 headless pipeline

Каталог содержит автономный smoke-pipeline для проверки вычислений без Orange GUI. Файлы `Var_Train.txt`, `Var_Test.txt`, `Var_Control.txt` в репозитории не найдены, поэтому `run_all.py` не заявляет результаты индивидуального варианта: он запускает только явно помеченный demo smoke-test и пишет `BLOCKED_BY_VARIANT_DATA.md`.

```bash
python run_all.py
pytest -q
```

Основной интерактивный интерфейс находится в `seminar_playground/`.
