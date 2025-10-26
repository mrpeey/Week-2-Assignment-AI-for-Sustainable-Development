@echo off
REM SmartFarm AI - Run API Tests
REM Make sure start_api.bat is running in another window first!

echo ========================================
echo SmartFarm AI - API Test Suite
echo ========================================
echo.
echo Testing endpoints...
echo.

python test_api.py

echo.
pause
