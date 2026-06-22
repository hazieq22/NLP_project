@echo off
cd /d "%~dp0"

echo Preparing GitHub upload for this NLP project...
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed or not available in PATH.
    echo Install Git for Windows first, then run this file again.
    pause
    exit /b 1
)

if not exist ".git" (
    git init
)

git branch -M main

git config user.name >nul 2>nul
if errorlevel 1 (
    set /p GIT_NAME=Enter your Git name: 
    git config user.name "%GIT_NAME%"
)

git config user.email >nul 2>nul
if errorlevel 1 (
    set /p GIT_EMAIL=Enter your GitHub email: 
    git config user.email "%GIT_EMAIL%"
)

git add -A

echo.
echo Files ready to upload:
git status --short
echo.

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Initial NLP sentiment analysis project"
) else (
    echo No new file changes to commit.
)

echo.
set /p REMOTE_URL=Paste your empty GitHub repository URL: 

git remote remove origin >nul 2>nul
git remote add origin "%REMOTE_URL%"
git push -u origin main

echo.
echo Done. If GitHub asks you to sign in, complete the browser login and rerun this file if needed.
pause
