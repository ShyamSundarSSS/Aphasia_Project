"""
Audio Output Diagnostic Tool
Tests if TTS and audio output are working correctly.
"""

import sys

print("="*60)
print("AUDIO DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check pyttsx3
print("\nTest 1: Checking pyttsx3 installation...")
try:
    import pyttsx3
    print("✓ pyttsx3 is installed")
except ImportError:
    print("✗ pyttsx3 is NOT installed")
    print("  Install with: pip install pyttsx3")
    sys.exit(1)

# Test 2: Initialize engine
print("\nTest 2: Initializing TTS engine...")
try:
    engine = pyttsx3.init()
    print("✓ TTS engine initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Test 3: Check voices
print("\nTest 3: Checking available voices...")
try:
    voices = engine.getProperty('voices')
    print(f"✓ Found {len(voices)} voices:")
    for i, voice in enumerate(voices):
        print(f"   {i+1}. {voice.name}")
except Exception as e:
    print(f"✗ Failed to get voices: {e}")

# Test 4: Check volume
print("\nTest 4: Checking volume settings...")
try:
    volume = engine.getProperty('volume')
    rate = engine.getProperty('rate')
    print(f"✓ Current volume: {volume * 100:.0f}%")
    print(f"✓ Current rate: {rate} WPM")
    
    # Set to maximum
    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 150)
    print("✓ Set volume to 100% and rate to 150 WPM")
except Exception as e:
    print(f"✗ Failed to set properties: {e}")

# Test 5: Speak test
print("\nTest 5: Speaking test...")
print("="*60)
print("TURN UP YOUR SPEAKERS NOW!")
print("You should hear: 'Hello. This is an audio test.'")
print("="*60)

input("Press ENTER when ready...")

try:
    test_text = "Hello. This is an audio test. Can you hear me? Testing one, two, three."
    print(f"\n🔊 Speaking: '{test_text}'")
    
    engine.say(test_text)
    engine.runAndWait()
    
    print("✓ Speech command completed")
    
    heard = input("\nDid you hear the voice? (yes/no): ").lower().strip()
    
    if heard == 'yes':
        print("\n✓✓✓ AUDIO IS WORKING! ✓✓✓")
        print("\nYour system audio is fine. The issue might be:")
        print("1. Sample generator GUI is not using the right settings")
        print("2. Try restarting the sample_generator.py")
    else:
        print("\n✗✗✗ AUDIO NOT WORKING ✗✗✗")
        print("\nPossible solutions:")
        print("1. Check if speakers are plugged in and turned ON")
        print("2. Increase COMPUTER VOLUME (Windows volume, not app)")
        print("3. Right-click speaker icon → Open Sound Settings")
        print("4. Make sure correct output device is selected")
        print("5. Try different speakers or headphones")
        print("6. Check Windows Sound Control Panel")
        print("7. Update audio drivers")
        
except Exception as e:
    print(f"✗ Speech failed: {e}")
    print("\nThis indicates a problem with your audio system.")
    print("Check Windows sound settings and audio drivers.")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
