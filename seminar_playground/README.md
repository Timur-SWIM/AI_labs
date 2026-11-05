# MLP Playground — ЛР №6

Локальный интерактивный тренажёр к ЛР6: один нейрон, градиентный спуск, настоящая NumPy MLP с паузой/продолжением и диагностические задания. Он не подключается к серверам, SSH, VPN, базам данных или внешним API. Все результаты живут в памяти вкладки браузера и экспортируются пользователем.

Математика автономно перенесена из `lab06/src/mlp.py` этого репозитория. Исходная ЛР6 не меняется. Соглашение: `x=rho*cos(phi)`, `y=rho*sin(phi)`.

## Запуск без Docker

macOS/Linux:

```bash
cd seminar_playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Windows PowerShell:

```powershell
cd seminar_playground
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Либо используйте `./run.sh` или `./run.ps1` после создания окружения. Откройте http://127.0.0.1:8501.

## Docker

```bash
cd seminar_playground
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
curl -f http://127.0.0.1:8501/_stcore/health
docker compose down
```

По умолчанию порт опубликован только на `127.0.0.1` Docker-хоста. Образ основан на официальном multi-arch `python:3.12-slim`; зависимости являются pure/Python wheels для обычных amd64/arm64. Проверяйте сборку на целевой архитектуре.

### Ubuntu VM → браузер физического хоста

`localhost` контейнера — это не localhost VM, а localhost VM — не localhost физического хоста. Оставьте закрытый compose-режим в VM и с физического хоста сделайте туннель (при настроенном SSH-доступе к собственной VM):

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@VM_IP
```

В Windows PowerShell встроенная команда такая же. При занятом порте: `ssh -N -L 18501:127.0.0.1:8501 USER@VM_IP`, затем откройте `http://127.0.0.1:18501`. Для VM за NAT нужны доступный адрес SSH или port forwarding гипервизора.

Менее приватный вариант только для собственной VM/LAN: `docker compose -f compose.yaml -f compose.lan.yaml up -d --build`, затем `http://VM_IP:8501`. Это открывает сервис на интерфейс VM: используйте bridged/host-only сеть или NAT forwarding, firewall и не публикуйте порт в интернет; после занятия вернитесь к обычному `docker compose down` и закрытому режиму.

## Проверки

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

`tests/` проверяет формулы, Adam pause/resume, нормализацию, экспорт и независимость сессий. `docker compose config -q` проверяет compose-описание.

См. также [STUDENT_GUIDE.md](STUDENT_GUIDE.md), [TEACHER_GUIDE.md](TEACHER_GUIDE.md), [EXERCISES.md](EXERCISES.md).
