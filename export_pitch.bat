@echo off
REM Export SmartFarm AI pitch deck to PowerPoint (.pptx)
REM Requires: Python and internet to install python-pptx once

setlocal ENABLEDELAYEDEXPANSION

where python >NUL 2>&1
IF ERRORLEVEL 1 (
  echo Python not found in PATH. Please install Python and retry.
  exit /b 1
)

REM Try to import python-pptx; install if missing
python -c "import pptx" >NUL 2>&1
IF ERRORLEVEL 1 (
  echo Installing python-pptx...
  pip install --user python-pptx || (
    echo Failed to install python-pptx. Install manually: pip install python-pptx
    exit /b 1
  )
)

python tools\build_pitch_pptx.py
IF ERRORLEVEL 1 (
  echo Failed to build PPTX. See errors above.
  exit /b 1
)

echo.
echo Pitch deck exported to docs\pitch\SmartFarm_AI_Pitch_Deck.pptx
exit /b 0
