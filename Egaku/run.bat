@echo off
setlocal
cd /d "%~dp0"
title Egaku - Image to Art Studio

if not exist ".venv\Scripts\python.exe" (
  echo Egaku environment not found. Run setup_gpu.bat first.
  pause
  exit /b 1
)

echo Checking Egaku source...
.venv\Scripts\python.exe -m py_compile main.py egaku\sketch_engine.py egaku\verify_runtime.py
if errorlevel 1 (
  echo.
  echo Python source validation failed. Run setup_gpu.bat or update the project.
  pause
  exit /b 1
)

echo.
echo Checking Egaku GPU runtime...
.venv\Scripts\python.exe egaku\verify_runtime.py
if errorlevel 1 (
  echo.
  echo Egaku GPU checks failed. Run setup_gpu.bat and check the NVIDIA driver.
  pause
  exit /b 1
)

echo.
echo Launching Egaku...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
  echo.
  echo Egaku closed with an error.
  echo Check %LOCALAPPDATA%\Egaku\logs\last-render-error.log if a render failed.
  pause
)
