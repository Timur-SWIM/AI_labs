# MLP Playground — ЛР №6 и ЛР №7

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

### Локальная сеть и Tailscale

Для этого компьютера подготовлен явный override: он публикует сервис только на LAN-адрес `192.168.10.50` и Tailscale-адрес `100.86.113.13`, не на всех интерфейсах.

```bash
docker compose -f compose.yaml -f compose.lan.yaml up -d --build
```

Открывайте приложение из локальной сети по `http://192.168.10.50:8501`, а с устройства в tailnet — по `http://100.86.113.13:8501` или `http://ai-dev.tail8780c6.ts.net:8501`. Доступ через Tailscale дополнительно подчиняется ACL tailnet. При изменении IP-адресов обновите `compose.lan.yaml`; закрытый режим возвращается командой `docker compose -f compose.yaml -f compose.lan.yaml down`.

Не публикуйте порт в интернет. Если нужен доступ из другой LAN/VM, используйте bridged/host-only сеть или NAT forwarding и ограничьте firewall.

## Проверки

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

`tests/` проверяет формулы, Adam pause/resume, нормализацию, экспорт и независимость сессий. `docker compose config -q` проверяет compose-описание.

См. также [STUDENT_GUIDE.md](STUDENT_GUIDE.md), [TEACHER_GUIDE.md](TEACHER_GUIDE.md), [EXERCISES.md](EXERCISES.md), [STUDENT_GUIDE_LAB07.md](STUDENT_GUIDE_LAB07.md), [TEACHER_GUIDE_LAB07.md](TEACHER_GUIDE_LAB07.md), [LAB07_IMPLEMENTATION_NOTES.md](LAB07_IMPLEMENTATION_NOTES.md). В приложении после вкладок ЛР6 доступен раздел «ЛР7 — прикладные задачи» с аппроксимацией, временным рядом и классификацией.
