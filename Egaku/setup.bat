@echo off
setlocal
cd /d "%~dp0"
title Egaku - OpenCV Graphite Setup

echo ==============================================
echo   EGAKU - OPENCV GRAPHITE FLOW SETUP
echo ==============================================
echo No AI model, CUDA or NVIDIA driver is required.
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python Launcher not found. Install 64-bit Python 3.12 first.
  goto failed
)
py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 (
  echo ERROR: Egaku requires 64-bit Python 3.12.
  goto failed
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m compileall -q main.py egaku
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m unittest discover -s tests -v
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import cv2,numpy,PIL,PySide6; print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__); print('OpenCV graphite pipeline: READY')"
if errorlevel 1 goto failed
echo.
echo ==============================================
echo   EGAKU OPENCV READY
echo ==============================================
echo Launch Egaku with run.bat.
pause
exit /b 0
:failed
echo.
echo Setup did not complete. Check the error above.
pause
exit /b 1
