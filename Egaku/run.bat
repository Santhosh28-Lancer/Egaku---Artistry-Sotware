@echo off
setlocal
cd /d "%~dp0"
title Egaku - Image to Art Studio
if not exist ".venv\Scripts\python.exe" (
  echo Egaku environment not found. Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m py_compile main.py egaku\sketch_engine.py
if errorlevel 1 (
  echo Python source validation failed. Run setup.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import cv2,numpy,PIL,PySide6; print('OpenCV:',cv2.__version__); print('Local graphite engine ready.')"
if errorlevel 1 (
  echo Egaku dependencies are missing. Run setup.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe main.py
if errorlevel 1 (
  echo Egaku closed with an error. Check %%LOCALAPPDATA%%\Egaku\logs\last-render-error.log if available.
  pause
)
