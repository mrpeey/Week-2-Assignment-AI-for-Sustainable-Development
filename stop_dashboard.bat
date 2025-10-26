@echo off
REM SmartFarm AI - Stop Dashboard

echo ========================================
echo SmartFarm AI - Stopping Dashboard
echo ========================================
echo Attempting to stop any running Streamlit dashboard instances...

REM Kill Streamlit processes launched for this app (matches dashboard.py)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process ^| Where-Object { $_.CommandLine -like '*-m streamlit run*dashboard.py*' } ^| ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }"

REM Fallback: also free common ports (8501, 8502)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports=@(8501,8502); foreach($p in $ports){Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess ^| Get-Unique ^| ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction Stop } catch {} }}"

echo Done. If a browser tab is open, you can close it now.
