@echo off
setlocal
cd /d "%~dp0"
title Egaku - Full Dependency and CUDA Setup

echo ==============================================
echo   EGAKU - FULL GPU ENVIRONMENT REPAIR
echo ==============================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python Launcher not found.
  echo Install 64-bit Python 3.12 and try again.
  goto failed
)

py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 (
  echo ERROR: Egaku requires 64-bit Python 3.12.
  goto failed
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python 3.12 virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)

.venv\Scripts\python.exe -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 (
  echo ERROR: The existing .venv is not 64-bit Python 3.12.
  echo Rename or delete the .venv folder, then run setup again.
  goto failed
)

echo.
echo [1/7] Updating pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed

echo.
echo [2/7] Installing CUDA-enabled PyTorch 2.13.0...
echo This uses the official PyTorch CUDA 13.2 wheel index.
.venv\Scripts\python.exe -m pip install "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple
if errorlevel 1 goto failed

echo.
echo [3/7] Installing the pinned Egaku dependencies...
.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto failed

echo.
echo [4/7] Checking Python package consistency...
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto failed

echo.
echo [5/7] Checking all Python source files...
.venv\Scripts\python.exe -m compileall -q main.py egaku
if errorlevel 1 goto failed

echo.
echo [6/7] Running Egaku unit tests...
.venv\Scripts\python.exe -m unittest discover -s tests -v
if errorlevel 1 goto failed

echo.
echo [7/7] Running PyTorch, CUDA, PEFT and GPU compute tests...
.venv\Scripts\python.exe -c "import torch,torchvision,diffusers,transformers,accelerate,peft,safetensors,huggingface_hub; from diffusers.utils.import_utils import is_peft_available; assert torch.__version__.split('+')[0] == '2.13.0', 'Unexpected PyTorch version'; assert torchvision.__version__.split('+')[0] == '0.28.0', 'torchvision does not match PyTorch 2.13'; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; assert torch.version.cuda is not None, 'CPU-only PyTorch detected'; assert is_peft_available(), 'Diffusers cannot detect PEFT'; x=torch.randn((256,256),device='cuda'); y=x@x; torch.cuda.synchronize(); p=torch.cuda.get_device_properties(0); assert torchvision.__version__.split('+')[0] == '0.28.0', 'torchvision version does not match PyTorch 2.13'; print('PyTorch:',torch.__version__); print('torchvision:',torchvision.__version__); print('CUDA runtime:',torch.version.cuda); print('Diffusers:',diffusers.__version__); print('Transformers:',transformers.__version__); print('PEFT:',peft.__version__); print('Accelerate:',accelerate.__version__); print('GPU:',p.name); print('VRAM: %.2f GB' % (p.total_memory/1024**3)); print('CUDA compute test: PASS')"
if errorlevel 1 (
  echo.
  echo ERROR: GPU runtime verification failed.
  echo Check your NVIDIA driver, then see the troubleshooting section in README.md.
  goto failed
)

echo.
echo ==============================================
echo   EGAKU ENVIRONMENT READY
echo ==============================================
echo Close any old Egaku window, then launch run.bat.
pause
exit /b 0

:failed
echo.
echo ==============================================
echo   SETUP DID NOT COMPLETE
echo ==============================================
echo Do not try rendering until this setup finishes successfully.
pause
exit /b 1
