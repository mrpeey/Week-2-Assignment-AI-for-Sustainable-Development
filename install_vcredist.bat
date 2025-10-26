@echo off
REM ============================================================================
REM TensorFlow Setup Helper - Downloads Visual C++ Redistributable
REM ============================================================================

echo.
echo ========================================================================
echo TensorFlow Setup Helper
echo ========================================================================
echo.
echo This script will help you download and install the required
echo Microsoft Visual C++ Redistributable for TensorFlow.
echo.

echo Step 1: Downloading VC++ Redistributable...
echo.
echo Download URL: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.

REM Try to download using PowerShell
powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe' -ErrorAction Stop}"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Download successful!
    echo.
    echo Step 2: Running installer...
    echo (This requires administrator privileges)
    echo.
    
    start /wait vc_redist.x64.exe
    
    echo.
    echo Step 3: Cleaning up...
    del vc_redist.x64.exe
    
    echo.
    echo ✓ Installation complete!
    echo.
    echo Step 4: Testing TensorFlow...
    echo.
    
    .venv-tf311\Scripts\python.exe -c "import tensorflow as tf; print(f'\n✓ TensorFlow {tf.__version__} is ready!\n')"
    
    if %ERRORLEVEL% EQU 0 (
        echo ========================================================================
        echo SUCCESS! TensorFlow is working correctly.
        echo ========================================================================
        echo.
        echo You can now run:
        echo   .venv-tf311\Scripts\streamlit run dashboard.py
        echo   .venv-tf311\Scripts\uvicorn api.main:app --reload
        echo.
    ) else (
        echo ========================================================================
        echo TensorFlow import failed. Please restart your computer and try again.
        echo ========================================================================
        echo.
    )
) else (
    echo.
    echo ✗ Download failed. Please download manually from:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo After installing, test with:
    echo   .venv-tf311\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__)"
    echo.
)

echo.
echo See TENSORFLOW_SETUP.md for more details.
echo.
pause
