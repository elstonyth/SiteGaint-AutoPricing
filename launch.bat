@echo off
title SiteGiant Pricing Automation
cd /d "%~dp0"

echo.
echo ============================================================
echo    SiteGiant Pricing Automation - Starting...
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Start the server
echo Starting web server at http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

:: Run uvicorn
.venv\Scripts\python.exe -m uvicorn src.webapp.main:app --host 127.0.0.1 --port 8000

pause
