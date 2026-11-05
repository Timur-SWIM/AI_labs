#!/usr/bin/env sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Не найден .venv. Создайте его: python3 -m venv .venv; .venv/bin/python -m pip install -r requirements.txt" >&2; exit 1
fi
exec "$PY" -m streamlit run "$HERE/app.py" --server.address 127.0.0.1 --server.port 8501
