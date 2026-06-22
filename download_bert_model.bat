@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_XET=1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" download_bert_model.py
) else (
    python download_bert_model.py
)

pause
