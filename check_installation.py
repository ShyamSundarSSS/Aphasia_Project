"""
Installation verification script for Aphasia Detection System.
Checks if all required packages are installed correctly.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - NOT FOUND")
        print(f"   Error: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg              - OK ({version_line})")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print(f"⚠️  FFmpeg              - NOT FOUND (will use librosa fallback)")
    return False

def main():
    print("="*60)
    print("APHASIA DETECTION SYSTEM - INSTALLATION CHECK")
    print("="*60)
    print()
    
    print("Core Dependencies:")
    print("-"*60)
    all_ok = True
    
    packages = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Librosa", "librosa"),
        ("SoundDevice", "sounddevice"),
        ("SoundFile", "soundfile"),
        ("PyTorch", "torch"),
        ("Whisper", "whisper"),
        ("Transformers", "transformers"),
        ("scikit-learn", "sklearn"),
        ("LightGBM", "lightgbm"),
        ("XGBoost", "xgboost"),
    ]
    
    for display_name, import_name in packages:
        if not check_package(display_name, import_name):
            all_ok = False
    
    print()
    print("External Dependencies:")
    print("-"*60)
    ffmpeg_ok = check_ffmpeg()
    
    print()
    print("="*60)
    
    if all_ok:
        print("✅ All Python packages are installed correctly!")
        if not ffmpeg_ok:
            print("⚠️  FFmpeg is missing but system will work with fallback")
            print("   To install FFmpeg: winget install ffmpeg")
    else:
        print("❌ Some packages are missing!")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("="*60)
    print()
    
    # Version information
    print("Version Information:")
    print("-"*60)
    try:
        import numpy
        print(f"NumPy:        {numpy.__version__}")
    except: pass
    
    try:
        import torch
        print(f"PyTorch:      {torch.__version__}")
    except: pass
    
    try:
        import whisper
        print(f"Whisper:      {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")
    except: pass
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except: pass
    
    print("="*60)
    print()
    print("🚀 System is ready to use!")
    print("   Run: python main.py")
    print()

if __name__ == "__main__":
    main()
