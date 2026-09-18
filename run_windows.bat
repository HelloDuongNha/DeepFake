@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Missing %PYTHON%
  echo Create a Windows virtual environment and install requirements.txt first.
  exit /b 1
)

set "DLC_PROFILE=windows-quality"
if not defined DLC_SOURCE_DIR set "DLC_SOURCE_DIR=%~dp0source_faces"
if not defined DLC_ENHANCER set "DLC_ENHANCER=GPEN-512"
if not defined DLC_ENHANCER_INTERVAL set "DLC_ENHANCER_INTERVAL=1"
if not defined DLC_MASK_BLUR set "DLC_MASK_BLUR=1.5"
if not defined DLC_MASK_EROSION set "DLC_MASK_EROSION=4"
if not defined DLC_DETAIL_STRENGTH set "DLC_DETAIL_STRENGTH=0.35"
if not defined DLC_FILM_GRAIN set "DLC_FILM_GRAIN=0.35"
if not defined DLC_HAIRLINE_GUARD set "DLC_HAIRLINE_GUARD=0.16"
if not defined DLC_COLOR_MATCH set "DLC_COLOR_MATCH=1"
if not defined DLC_BLEND_MODE set "DLC_BLEND_MODE=poisson"
if not defined DLC_CAPTURE_WIDTH set "DLC_CAPTURE_WIDTH=1280"
if not defined DLC_CAPTURE_HEIGHT set "DLC_CAPTURE_HEIGHT=720"
if not defined DLC_PREFER_TENSORRT set "DLC_PREFER_TENSORRT=0"
set "PYTHONUNBUFFERED=1"

where nvidia-smi >nul 2>&1
if errorlevel 1 echo NVIDIA driver tools not found. Provider auto-detection will use an available fallback.

"%PYTHON%" run.py --execution-provider auto --frame-processor face_swapper --live-resizable %*
exit /b %errorlevel%
