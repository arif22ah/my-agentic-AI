@echo off
REM Self-Healing Code Generator - Clean Windows CMD Launcher (v3)
echo =========================================================
echo   Self-Healing Code Generator (v3) - Clean Launcher
echo =========================================================

if "%~1"=="test" goto :test_mode
if "%~1"=="local" goto :local_mode

where docker >nul 2>nul
if errorlevel 1 goto :local_mode

docker info >nul 2>nul
if errorlevel 1 goto :local_mode

echo [1/3] Stopping any old containers...
docker compose down --remove-orphans

echo [2/3] Building sandbox image - selfheal-sandbox:latest...
docker build -t selfheal-sandbox:latest sandbox

echo [3/3] Building and starting API and DB containers...
docker compose up --build --force-recreate
goto :eof

:test_mode
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -e ".[dev,local]"
pytest -q
goto :eof

:local_mode
echo [INFO] Running in local Windows CMD mode (SANDBOX_MODE=local)...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -e ".[dev,local]"
set SANDBOX_MODE=local
set DATABASE_URL=
echo [INFO] Starting FastAPI + Live Web UI on http://localhost:8000 ...
uvicorn my_agent.api:app --host 0.0.0.0 --port 8000 --reload
