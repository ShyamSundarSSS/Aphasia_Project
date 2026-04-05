@echo off
echo ============================================================
echo FFmpeg Installation Helper for Aphasia Detection System
echo ============================================================
echo.
echo FFmpeg is required for Whisper to process audio files.
echo.
echo Option 1: Install via winget (Windows 10+)
echo   Run: winget install ffmpeg
echo.
echo Option 2: Install via Chocolatey
echo   Run: choco install ffmpeg
echo.
echo Option 3: Manual Installation
echo   1. Download from: https://www.gyan.dev/ffmpeg/builds/
echo   2. Extract the zip file
echo   3. Add the 'bin' folder to your system PATH
echo.
echo Checking if FFmpeg is already installed...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [NOT FOUND] FFmpeg is not installed or not in PATH
    echo.
    echo Would you like to install FFmpeg via winget? (Requires Windows 10+)
    set /p choice="Enter Y to install, N to cancel: "
    if /i "%choice%"=="Y" (
        echo.
        echo Installing FFmpeg via winget...
        winget install ffmpeg
        echo.
        echo Installation complete! Please restart your terminal.
    ) else (
        echo.
        echo Please install FFmpeg manually and add it to your PATH.
        echo See: https://www.wikihow.com/Install-FFmpeg-on-Windows
    )
) else (
    echo.
    echo [FOUND] FFmpeg is already installed!
    ffmpeg -version
)
echo.
pause
