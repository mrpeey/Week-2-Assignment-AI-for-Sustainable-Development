@echo off
REM Create a Python 3.11 virtual environment and install TensorFlow (CPU)
REM Usage: double-click this file or run in PowerShell/CMD

setlocal ENABLEDELAYEDEXPANSION

REM Check for Python 3.11 via the Windows launcher
py -3.11 -V >NUL 2>&1
IF ERRORLEVEL 1 (
  echo.
  echo Python 3.11 is required to install TensorFlow (your current Python is newer).
  echo.
  echo Please install Python 3.11 from:
  echo   https://www.python.org/downloads/release/python-3110/
  echo Or use Miniconda and run:
  echo   conda create -n tf311 python=3.11 -y ^&^& conda activate tf311 ^&^& pip install -r requirements-tf.txt
  echo.
  pause
  exit /b 1
)

set VENV_DIR=.venv-tf311

if exist %VENV_DIR% (
  echo Virtual environment already exists at %VENV_DIR%
) else (
  echo Creating virtual environment with Python 3.11 at %VENV_DIR% ...
  py -3.11 -m venv %VENV_DIR%
  IF ERRORLEVEL 1 (
    echo Failed to create virtual environment with Python 3.11.
    exit /b 1
  )
)

call %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
  echo Failed to activate virtual environment.
  exit /b 1
)

python -m pip install --upgrade pip
IF EXIST requirements-tf.txt (
  pip install -r requirements-tf.txt
) ELSE (
  pip install tensorflow==2.16.1
)

echo.
echo TensorFlow environment ready.
echo To activate later, run:
echo   %VENV_DIR%\Scripts\activate.bat

echo.
echo Tip: Run your dashboard/API using this environment to enable CNN/RL modules.
endlocal
exit /b 0
