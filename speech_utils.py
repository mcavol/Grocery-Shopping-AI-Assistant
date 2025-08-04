"""
Speech recognition utilities using OpenAI Whisper with WORKING audio recording.
FIXED implementation with automatic transcription and proper integration.
"""

import whisper
import tempfile
import os
import streamlit as st
import logging
import numpy as np
from typing import Optional
import io

logger = logging.getLogger(__name__)


@st.cache_resource
def load_whisper_model(model_size: str = "base"):
    """Load Whisper model with caching."""
    try:
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logger.info("✅ Whisper model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        st.error(f"❌ Failed to load speech recognition model: {e}")
        return None


def transcribe_audio(audio_data: bytes, model_size: str = "base") -> Optional[str]:
    """
    Transcribe audio data using Whisper.
    
    Args:
        audio_data: Audio data in bytes
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        Transcribed text or None if failed
    """
    try:
        # Load Whisper model
        model = load_whisper_model(model_size)
        if model is None:
            return None
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe audio
            logger.info("🎤 Transcribing audio...")
            result = model.transcribe(temp_audio_path, language="en")
            transcribed_text = result["text"].strip()
            
            logger.info(f"✅ Transcription successful: {transcribed_text[:100]}...")
            return transcribed_text
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        st.error(f"❌ Speech transcription failed: {e}")
        return None


def get_whisper_model_info():
    """Get information about available Whisper models."""
    models = {
        "tiny": "39 MB - Fastest, least accurate",
        "base": "74 MB - Good balance (recommended)",
        "small": "244 MB - Better accuracy",
        "medium": "769 MB - High accuracy",
        "large": "1550 MB - Best accuracy"
    }
    return models


def create_voice_input_interface(model_size: str = "base") -> Optional[str]:
    """
    Create a complete voice input interface with automatic transcription.
    Returns transcribed text or None.
    """
    
    try:
        from audio_recorder_streamlit import audio_recorder
        
        st.info("🎤 **Click the microphone button to record your voice**")
        st.caption("📝 Speak clearly and the recording will automatically stop after 3 seconds of silence.")
        
        # Use audio_recorder_streamlit with automatic processing
        audio_bytes = audio_recorder(
            text="",  # No text to make button cleaner
            recording_color="#e74c3c",  # Red when recording
            neutral_color="#3498db",    # Blue when not recording
            icon_name="microphone",     # Microphone icon
            icon_size="2x",             # Icon size
            pause_threshold=3.0,        # Auto-stop after 3 seconds of silence
            sample_rate=16000,          # Sample rate for Whisper
            key="voice_input_recorder"  # Unique key
        )
        
        # If we got audio data, automatically transcribe it
        if audio_bytes is not None:
            # Check if this is new audio data
            if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != audio_bytes:
                st.session_state.last_processed_audio = audio_bytes
                
                st.success("✅ Audio recorded! Transcribing...")
                
                # Show audio player for confirmation
                st.audio(audio_bytes, format="audio/wav")
                
                # Automatically transcribe
                with st.spinner(f"🔄 Transcribing speech using {model_size} model..."):
                    try:
                        transcribed_text = transcribe_audio(audio_bytes, model_size)
                        
                        if transcribed_text and transcribed_text.strip():
                            st.success(f"🎯 **Transcribed**: {transcribed_text}")
                            logger.info(f"✅ Voice input transcribed: {transcribed_text}")
                            return transcribed_text.strip()
                        else:
                            st.error("❌ Could not understand speech. Please try again or speak more clearly.")
                            st.info("💡 Tips: Speak clearly, avoid background noise, ensure microphone is working")
                            return None
                            
                    except Exception as e:
                        st.error(f"❌ Speech recognition failed: {e}")
                        st.info("💡 Try speaking more clearly or check your microphone")
                        return None
        
        return None
        
    except ImportError:
        st.error("❌ **Audio recording package not installed**")
        st.code("pip install audio-recorder-streamlit")
        st.info("💡 Install the audio-recorder-streamlit package to enable voice input")
        return None
    except Exception as e:
        logger.error(f"Audio recorder error: {e}")
        st.error(f"❌ Audio recording failed: {e}")
        st.info("💡 Make sure microphone permissions are enabled in your browser")
        return None


def create_webrtc_recorder():
    """Fallback to WebRTC recorder if available."""
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import av
        
        st.warning("📦 Using WebRTC fallback for audio recording")
        
        class AudioProcessor:
            def __init__(self):
                self.audio_frames = []
            
            def recv(self, frame):
                sound = frame.to_ndarray()
                self.audio_frames.append(sound)
                return frame
        
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        audio_processor = AudioProcessor()
        
        webrtc_ctx = webrtc_streamer(
            key="voice-recorder-webrtc",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": False, "audio": True},
            audio_frame_callback=audio_processor.recv,
        )
        
        if webrtc_ctx.audio_receiver:
            st.info("🔴 Recording... Click 'STOP' when done speaking")
        
        if st.button("🛑 Stop and Process Audio", key="stop_webrtc"):
            if audio_processor.audio_frames:
                # Convert to audio bytes
                audio_data = np.concatenate(audio_processor.audio_frames, axis=0)
                
                # Convert to WAV format
                import wave
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                wav_buffer.seek(0)
                audio_bytes = wav_buffer.getvalue()
                
                st.success("✅ Audio processed!")
                st.audio(audio_bytes, format="audio/wav")
                
                # Auto-transcribe
                with st.spinner("🔄 Transcribing..."):
                    transcribed_text = transcribe_audio(audio_bytes, "base")
                    if transcribed_text:
                        st.success(f"🎯 **Transcribed**: {transcribed_text}")
                        return transcribed_text
                    else:
                        st.error("❌ Transcription failed")
                        return None
            else:
                st.warning("⚠️ No audio captured. Try again.")
                return None
        
        return None
        
    except ImportError:
        return create_file_upload_fallback()


def create_file_upload_fallback():
    """Final fallback - file upload."""
    st.warning("🎤 **Audio recording not available** - Using file upload fallback")
    st.info("💡 Record audio on your device and upload it below:")
    
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Record audio using your device's voice recorder app, then upload here",
        key="audio_upload"
    )
    
    if uploaded_file is not None:
        st.success("📁 Audio file uploaded!")
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')
        
        # Auto-transcribe uploaded file
        with st.spinner("🔄 Transcribing uploaded audio..."):
            transcribed_text = transcribe_audio(uploaded_file.getvalue(), "base")
            if transcribed_text:
                st.success(f"🎯 **Transcribed**: {transcribed_text}")
                return transcribed_text
            else:
                st.error("❌ Transcription failed")
                return None
    
    return None


def create_best_voice_input(model_size: str = "base") -> Optional[str]:
    """Create the best available voice input with automatic transcription."""
    
    # Check what packages are available
    packages_status = check_audio_packages()
    
    if packages_status["audio_recorder_streamlit"]:
        return create_voice_input_interface(model_size)
    elif packages_status["streamlit_webrtc"]:
        st.info("🎤 Using streamlit-webrtc (Fallback)")
        return create_webrtc_recorder()
    else:
        st.warning("📦 No audio recording packages found")
        return create_file_upload_fallback()


def check_audio_packages():
    """Check which audio packages are available."""
    status = {
        "audio_recorder_streamlit": False,
        "streamlit_webrtc": False,
        "whisper": False
    }
    
    try:
        import audio_recorder_streamlit
        status["audio_recorder_streamlit"] = True
    except ImportError:
        pass
    
    try:
        import streamlit_webrtc
        status["streamlit_webrtc"] = True
    except ImportError:
        pass
    
    try:
        import whisper
        status["whisper"] = True
    except ImportError:
        pass
    
    return status


def show_audio_setup_instructions():
    """Show setup instructions for audio packages."""
    st.markdown("### 🔧 Audio Setup Instructions")
    
    packages = check_audio_packages()
    
    if not packages["audio_recorder_streamlit"]:
        st.markdown("**📦 Install audio recording (Required):**")
        st.code("pip install audio-recorder-streamlit")
    
    if not packages["whisper"]:
        st.markdown("**🤖 Install speech recognition (Required):**")
        st.code("pip install openai-whisper")
    
    if packages["audio_recorder_streamlit"] and packages["whisper"]:
        st.success("✅ All audio packages are installed!")
    
    st.markdown("**🌐 Browser Requirements:**")
    st.info("• Allow microphone access when prompted\n• Use HTTPS in production\n• Chrome, Firefox, Safari, or Edge recommended")
    
    return packages["audio_recorder_streamlit"] and packages["whisper"]


def install_audio_packages_button():
    """Show button to install audio packages."""
    if st.button("📦 Install Audio Packages", help="Install required packages for voice input", key="install_audio"):
        try:
            import subprocess
            import sys
            
            with st.spinner("Installing audio packages..."):
                # Install audio-recorder-streamlit
                result1 = subprocess.run([
                    sys.executable, "-m", "pip", "install", "audio-recorder-streamlit"
                ], capture_output=True, text=True)
                
                # Install openai-whisper  
                result2 = subprocess.run([
                    sys.executable, "-m", "pip", "install", "openai-whisper"
                ], capture_output=True, text=True)
                
                if result1.returncode == 0 and result2.returncode == 0:
                    st.success("✅ Audio packages installed! Please restart the app.")
                    st.info("🔄 Restart the Streamlit app to use voice input")
                else:
                    st.error("❌ Installation failed. Install manually:")
                    st.code("pip install audio-recorder-streamlit openai-whisper")
                    
        except Exception as e:
            st.error(f"❌ Installation error: {e}")
            st.info("💡 Install manually with: pip install audio-recorder-streamlit openai-whisper")


def test_audio_recorder():
    """Test function to verify audio recorder works correctly."""
    st.subheader("🧪 Audio Recorder Test")
    
    packages = check_audio_packages()
    
    if not packages["audio_recorder_streamlit"]:
        st.error("❌ audio-recorder-streamlit not installed")
        return False
    
    if not packages["whisper"]:
        st.error("❌ openai-whisper not installed")
        return False
    
    st.success("✅ All packages installed")
    
    # Test the recorder
    try:
        transcribed_text = create_voice_input_interface("tiny")  # Use fastest model for testing
        
        if transcribed_text:
            st.success(f"✅ Voice input test successful: '{transcribed_text}'")
            return True
        else:
            st.info("⏳ Record some audio to test the system")
            
    except Exception as e:
        st.error(f"❌ Voice input test failed: {e}")
        return False
    
    return True