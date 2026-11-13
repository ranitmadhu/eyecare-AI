@echo off
title EyeCare AI - FastAPI Server
echo ========================================================
echo        Starting EyeCare AI FastAPI Server
echo ========================================================
echo.

set VENV_PY=C:\Users\Ranit Madhu\Desktop\fast api\myvenv\Scripts\python.exe

if exist "%VENV_PY%" (
    echo Using Virtual Environment: %VENV_PY%
    "%VENV_PY%" -m uvicorn app:app --reload --port 5000
) else (
    echo Using System Python:
    python -m uvicorn app:app --reload --port 5000
)

pause
