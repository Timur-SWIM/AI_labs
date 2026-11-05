$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "Не найден .venv. Выполните: py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"; exit 1 }
& $python -m streamlit run (Join-Path $root "app.py") --server.address 127.0.0.1 --server.port 8501
