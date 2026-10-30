@echo off
setlocal
cd /d "%~dp0"
title Egaku - Image to Art Studio

if not exist ".venv\Scripts\python.exe" (
  echo Egaku environment not found. Run setup_gpu.bat first.
  pause
  exit /b 1
)

echo Checking Egaku runtime...
.venv\Scripts\python.exe -c "import torch,diffusers,accelerate,peft; from diffusers.utils.import_utils import is_peft_available; assert torch.cuda.is_available() and torch.version.cuda is not None, 'CUDA PyTorch is not available'; assert is_peft_available(), 'PEFT is not detected'; print('GPU:',torch.cuda.get_device_name(0)); print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda); print('Diffusers:',diffusers.__version__,'PEFT:',peft.__version__)"
if errorlevel 1 (
  echo.
  echo Egaku runtime verification failed. Run setup_gpu.bat and check the NVIDIA driver.
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m py_compile main.py egaku\sketch_engine.py
if errorlevel 1 (
  echo.
  echo Python source validation failed. Do not render; update the project files.
  pause
  exit /b 1
)

.venv\Scripts\python.exe main.py
if errorlevel 1 (
  echo.
  echo Egaku closed with an error. See %LOCALAPPDATA%\Egaku\logs\last-render-error.log if a render failed.
  pause
)
