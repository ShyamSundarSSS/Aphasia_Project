@echo off
echo ============================================================
echo COMPLETE SETUP - Aphasia Detection System
echo ============================================================
echo.
echo This script will:
echo   1. Create a virtual environment
echo   2. Install all required packages
echo   3. Check for FFmpeg
echo   4. Verify the installation
echo.
pause

REM Step 1: Create virtual environment
echo.
echo ============================================================
echo Creating Virtual Environment...
echo ============================================================
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Step 2: Activate and install
call venv\Scripts\activate.bat

echo.
echo ============================================================
echo Upgrading pip...
echo ============================================================
python -m pip install --upgrade pip --quiet

echo.
echo ============================================================
echo Installing dependencies... (this will take several minutes)
echo ============================================================
echo.
echo [1/7] Installing NumPy and SciPy...
pip install numpy scipy --quiet

echo [2/7] Installing audio processing libraries...
pip install librosa sounddevice soundfile audioread resampy --quiet

echo [3/7] Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo [4/7] Installing Whisper...
pip install openai-whisper --quiet

echo [5/7] Installing Transformers...
pip install transformers tokenizers --quiet

echo [6/7] Installing ML libraries...
pip install scikit-learn lightgbm xgboost --quiet

echo [7/7] Installing utilities...
pip install tqdm --quiet

echo.
echo ============================================================
echo Checking FFmpeg...
echo ============================================================
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] FFmpeg not found!
    echo.
    echo Attempting to install FFmpeg via winget...
    winget install ffmpeg --silent
    if errorlevel 1 (
        echo.
        echo Could not install FFmpeg automatically.
        echo The system will work with librosa fallback.
        echo.
        echo To install FFmpeg manually:
        echo   1. Download from: https://www.gyan.dev/ffmpeg/builds/
        echo   2. Extract and add to PATH
        echo   3. Restart your terminal
    ) else (
        echo FFmpeg installed successfully!
        echo Please restart your terminal for changes to take effect.
    )
) else (
    echo [SUCCESS] FFmpeg is installed!
)

echo.
echo ============================================================
echo Verifying Installation...
echo ============================================================
python -c "import torch; import whisper; import transformers; import librosa; print('All core packages imported successfully!')"
if errorlevel 1 (
    echo.
    echo [ERROR] Some packages failed to import
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Virtual environment created at: venv\
echo.
echo To activate the environment:
echo   venv\Scripts\activate
echo.
echo To run the application:
echo   python main.py
echo.
echo To see all options:
echo   python main.py --help
echo.
echo To list scenarios:
echo   python main.py --list-scenarios
echo.
pause
