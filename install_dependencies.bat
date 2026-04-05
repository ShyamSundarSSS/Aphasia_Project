@echo off
echo ============================================================
echo Installing Dependencies for Aphasia Detection System
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
)

echo.
echo ============================================================
echo Step 1: Upgrading pip
echo ============================================================
python -m pip install --upgrade pip setuptools wheel

echo.
echo ============================================================
echo Step 2: Installing PyTorch (this may take a while)
echo ============================================================
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo ============================================================
echo Step 3: Installing core audio processing libraries
echo ============================================================
pip install numpy scipy librosa sounddevice soundfile audioread resampy

echo.
echo ============================================================
echo Step 4: Installing Whisper and Transformers
echo ============================================================
pip install openai-whisper transformers tokenizers

echo.
echo ============================================================
echo Step 5: Installing machine learning libraries
echo ============================================================
pip install scikit-learn lightgbm xgboost

echo.
echo ============================================================
echo Step 6: Installing utilities
echo ============================================================
pip install tqdm

echo.
echo ============================================================
echo Step 7: Checking FFmpeg Installation
echo ============================================================
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] FFmpeg is not installed!
    echo.
    echo FFmpeg is required for optimal audio processing with Whisper.
    echo The system will use librosa as fallback, but FFmpeg is recommended.
    echo.
    echo To install FFmpeg, choose one option:
    echo   1. Run: winget install ffmpeg
    echo   2. Run: choco install ffmpeg
    echo   3. Manual: https://www.gyan.dev/ffmpeg/builds/
    echo.
    set /p install_ffmpeg="Would you like to try installing FFmpeg via winget now? (Y/N): "
    if /i "%install_ffmpeg%"=="Y" (
        echo Installing FFmpeg...
        winget install ffmpeg
        echo.
        echo Please restart your terminal after installation completes.
    )
) else (
    echo [SUCCESS] FFmpeg is already installed!
    ffmpeg -version | findstr "version"
)

echo.
echo ============================================================
echo Installation Summary
echo ============================================================
echo.
echo Checking installed packages...
pip list | findstr "torch whisper transformers librosa sounddevice"

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo To run the application:
echo   python main.py
echo.
echo For help:
echo   python main.py --help
echo.
pause
