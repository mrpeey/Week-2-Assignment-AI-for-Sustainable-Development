@echo off
REM Quick launcher for SmartFarm AI Dashboard
REM Runs the Streamlit dashboard on localhost:8501

echo ============================================================
echo SMARTFARM AI DASHBOARD
echo ============================================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

C:\Python314\python.exe -m streamlit run src\dashboard.py --server.port 8501 --server.headless true

pause
