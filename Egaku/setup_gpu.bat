@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Install Python 3.12 first.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
pause
