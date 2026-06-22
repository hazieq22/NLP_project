@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
  echo Streamlit environment not found.
  echo Please run setup first or ask Codex to recreate the virtual environment.
  pause
  exit /b 1
)

echo Starting Movie Review Sentiment Detector...
echo.
".venv\Scripts\streamlit.exe" run app.py
pause
