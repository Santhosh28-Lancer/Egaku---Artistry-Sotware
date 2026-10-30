@echo off
setlocal
cd /d "%~dp0"
echo Egaku now uses local OpenCV CPU processing. CUDA and AI models are no longer required.
echo Starting the compatibility setup...
call setup.bat
