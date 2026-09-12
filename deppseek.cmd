@echo off
REM DeppSeek launcher. Prefers the project virtual environment, falls back to
REM whatever python is on PATH so the command still works before setup has run.
setlocal
set "DEPPSEEK_HOME=%~dp0"
set "VENV_PY=%DEPPSEEK_HOME%.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -m deppseek.cli %*
) else (
    echo [deppseek] No virtual environment found. Run setup.ps1 first. 1>&2
    echo [deppseek] Trying the system Python. 1>&2
    python -m deppseek.cli %*
)
endlocal
