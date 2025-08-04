# 🎤 Audio Recording Setup Guide

This guide will help you set up working speech recognition for the Grocery Shopping AI Assistant.

## 🔧 Installation Steps

### 1. Install Required Packages

```bash
# Install the updated requirements
pip install -r requirements.txt

# Or install audio packages individually
pip install audio-recorder-streamlit
pip install streamlit-webrtc
pip install pyaudio
```

### 2. Platform-Specific Setup

#### **Windows:**
```bash
# If PyAudio installation fails, install via conda:
conda install pyaudio

# Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

#### **macOS:**
```bash
# Install portaudio first
brew install portaudio

# Then install PyAudio
pip install pyaudio
```

#### **Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio

# Then install PyAudio
pip install pyaudio
```

### 3. Browser Requirements

For the best speech recognition experience:

- **Chrome/Chromium**: ✅ Full support
- **Firefox**: ✅ Good support  
- **Safari**: ⚠️ Limited support
- **Edge**: ✅ Good support

**Note**: Speech recognition requires HTTPS or localhost for security reasons.

## 🎯 Available Speech Input Methods

The app will automatically detect and use the best available method:

### Method 1: audio-recorder-streamlit (Recommended)
- ✅ Simple one-click recording
- ✅ Works in most browsers
- ✅ No complex setup

### Method 2: streamlit-webrtc (Advanced)
- ✅ Real-time audio streaming
- ✅ Better quality
- ⚠️ Requires WebRTC support

### Method 3: File Upload (Fallback)
- ✅ Always works
- 📁 Upload pre-recorded audio files
- 🎤 Record on your device first

## 🚀 Quick Test

1. Start the application:
```bash
streamlit run main.py
```

2. Enable voice input in the sidebar
3. Look for the "🎤 Voice Input" section
4. Click the microphone button to test

## 🐛 Troubleshooting

### "streamlit-audiorec not available"
- ✅ **Fixed**: We replaced this with working alternatives

### "Permission denied" for microphone
- Check browser permissions (click 🔒 next to URL)
- Allow microphone access for the site

### "PyAudio not found"
- Follow platform-specific installation above
- Try conda instead of pip on Windows

### Recording button not appearing
- Check browser console for errors (F12)
- Try refreshing the page
- Ensure you're on HTTPS or localhost

### Audio quality issues
- Try different Whisper models in sidebar
- Speak clearly and close to microphone
- Reduce background noise

## 📱 Mobile Usage

The app works on mobile browsers:
- **iOS Safari**: Use file upload method
- **Android Chrome**: Voice recording should work
- **Mobile Chrome**: Good support for WebRTC

## 🔧 Development Notes

If you're modifying the audio code:

1. The app automatically detects available packages
2. Falls back gracefully if packages are missing  
3. File upload is always available as last resort
4. All audio processing happens client-side

## 🆘 Still Having Issues?

1. Check the browser console (F12 → Console)
2. Verify microphone works in other apps
3. Try the file upload method as alternative
4. Check that you're using HTTPS or localhost

## 🎉 Success Indicators

When working correctly, you should see:
- ✅ "Enable Voice Input" checkbox in sidebar
- 🎤 Microphone button in Voice Input section  
- 🔄 "Transcribe Audio" button after recording
- 🎯 Transcribed text appears after processing

The speech recognition uses OpenAI Whisper for high-quality transcription, so it works offline once the model is downloaded!
