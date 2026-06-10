@echo off
echo ================================
echo AI-Agent-Qa Windows Quick Start
echo ================================

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo If this is your first run, copy .env.example to .env and fill DEEPSEEK_API_KEY.
echo.
echo Starting Streamlit...
streamlit run src/streamlit_app.py
