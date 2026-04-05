@echo off
echo ============================================================
echo Creating Virtual Environment for Aphasia Detection System
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Virtual environment created successfully!
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate
echo.
pause
