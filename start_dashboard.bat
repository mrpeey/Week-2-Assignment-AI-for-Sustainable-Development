@echo off
REM SmartFarm AI - Start Dashboard
REM Launches the Streamlit web interface

echo ========================================
echo SmartFarm AI - Starting Dashboard
echo ========================================
echo.
echo Installing dependencies if needed...
pip install -q streamlit plotly pandas numpy scikit-learn matplotlib seaborn pillow
echo.
echo Starting dashboard...
echo The dashboard will open in your browser automatically
echo.

REM Suppress onboarding/usage stats prompt for a clean start
set STREAMLIT_BROWSER_GATHERUSAGESTATS=false

REM Launch the Streamlit app
streamlit run src\dashboard.py
