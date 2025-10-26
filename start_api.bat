@echo off
REM SmartFarm AI - Start API Server
REM Run this in one terminal window

echo ========================================
echo SmartFarm AI - Starting API Server
echo ========================================
echo.
echo The API will be available at:
echo   http://127.0.0.1:8000
echo.
echo API Documentation:
echo   http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
