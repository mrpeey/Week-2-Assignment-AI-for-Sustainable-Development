@echo off
REM Quick launcher for SmartFarm AI FastAPI Backend
REM Runs the API server on localhost:8000

echo ============================================================
echo SMARTFARM AI API SERVER
echo ============================================================
echo.
echo Starting FastAPI server...
echo API will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Install uvicorn if needed
C:\Python314\python.exe -m pip install uvicorn --quiet --user

REM Start the API server
C:\Python314\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause
