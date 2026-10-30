@echo off
setlocal
cd /d "%~dp0"
title Egaku Local AI Pencil - Setup
echo ==============================================
echo   EGAKU - Local AI Pencil Setup
echo ==============================================
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
.venv\Scripts\python.exe -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
  echo Installing CUDA-enabled PyTorch 2.7.1 for CUDA 11.8 runtime...
  .venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
if errorlevel 1 goto failed
echo.
echo Setup completed. Run run.bat to launch Egaku.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
