# MLP Playground — ЛР6 и ЛР7

Локальное учебное приложение на Streamlit. Главная страница ведёт в ЛР6, ЛР7 или справку; результаты живут только в текущей браузерной сессии.

## Запуск без Docker

macOS/Linux:

```bash
cd seminar_playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Windows PowerShell:

```powershell
cd seminar_playground
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Откройте http://127.0.0.1:8501.

## Docker: закрытый режим по умолчанию

```bash
cd seminar_playground
docker compose build
docker compose up -d
docker compose ps
curl -f http://127.0.0.1:8501/_stcore/health
docker compose down
```

`compose.yaml` публикует порт только на `127.0.0.1`. Образ работает без root, с read-only filesystem, tmpfs и лимитами ресурсов.

### Ubuntu VM → браузер физического компьютера

Оставьте закрытый compose-режим в VM и используйте SSH-туннель:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@VM_IP
```

Затем откройте http://127.0.0.1:8501 на физическом компьютере.

### Опциональный LAN/Tailscale override

`compose.lan.yaml` — только шаблон для доверенной сети. Он не содержит персональных IP по умолчанию. Перед запуском укажите фактические адреса интерфейсов:

```bash
LAN_BIND_IP=192.168.x.y TAILSCALE_BIND_IP=100.x.y.z \
  docker compose -f compose.yaml -f compose.lan.yaml up -d --build
```

Проверьте `ip -brief address` и `tailscale ip -4`. Доступ также зависит от firewall и Tailscale ACL. Не публикуйте сервис в интернет.

## Лаборатории

- **ЛР6 · Как обучается сеть** — четыре последовательных упражнения: один нейрон, градиентный спуск, живая MLP, диагностика.
- **ЛР7 · Как выбрать модель** — мастер из пяти этапов: данные, постановка, обучение, сравнение, Orange.
- **Теория и справка** — Train/Test/Control, Loss, GF/BF, overfitting и Orange.

## Проверки

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

См. [STUDENT_GUIDE.md](STUDENT_GUIDE.md), [TEACHER_GUIDE.md](TEACHER_GUIDE.md), [UX_REVIEW.md](UX_REVIEW.md), [LAB07_IMPLEMENTATION_NOTES.md](LAB07_IMPLEMENTATION_NOTES.md).

## ЛР7: сравнение моделей

Маршрут: «Данные и постановка» → «Эксперименты» → «Выводы и Orange». Зафиксируйте Train/Test/Control и признаки, обучите и сохраните А, нажмите «Новый эксперимент», обучите и сохраните Б. История привязана к данным и не очищается при смене архитектуры. По Test выберите сохранённую модель, зафиксируйте её и только затем откройте Control. ZIP и CSV относятся к выбранной модели. Экспорт JSON позволяет перенести историю между браузерными сессиями; без него история может исчезнуть.
