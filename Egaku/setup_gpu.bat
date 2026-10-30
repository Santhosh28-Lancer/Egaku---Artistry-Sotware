@echo off
setlocal
cd /d "%~dp0"
title Egaku - Setup
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.12 first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import peft; from diffusers.utils.import_utils import is_peft_available; assert is_peft_available(); print('PEFT backend:', peft.__version__)"
if errorlevel 1 (
  echo PEFT backend verification failed. Re-run this setup or install peft manually.
  goto failed
)
echo.
echo Egaku dependencies are installed.
echo Close any open Egaku window, then run run.bat.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
