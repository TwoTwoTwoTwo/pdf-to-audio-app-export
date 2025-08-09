import streamlit as st
# Ensure XTTS import hook is installed before importing TTS
try:
    import import_hook_xtts_patch  # installs hook on import
    print("🔧 XTTS import hook ensured in voice_editor")
except Exception:
    pass
from TTS.api import TTS
import tempfile
import os
import json
from typing import Dict, Optional, List
from pathlib import Path
import torch

# Use the final working XTTS implementation
try:
    from xtts_final_working import get_final_working_tts, generate_tts_audio
    print("✅ Using final working XTTS implementation")
    XTTS_WORKING = True
except ImportError:
    print("❌ Final working XTTS implementation not found")
    XTTS_WORKING = False
    get_final_working_tts = None
    generate_tts_audio = None

try:
    import yt_dlp
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

class YouTubeCaptionExtractor:
    """Extract captions and speaker information from YouTube videos"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_captions(self, video_url: str) -> Optional[Dict]:
        """Get captions from YouTube video"""
        if not YOUTUBE_AVAILABLE:
            return None
            
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return None
        
        try:
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Prefer manually created transcripts over auto-generated
            transcript = None
            for t in transcript_list:
                if not t.is_generated:
                    transcript = t
                    break
            
            if not transcript:
                # Fall back to auto-generated if available
                transcript = transcript_list.find_generated_transcript(['en'])
            
            if transcript:
                captions = transcript.fetch()
                return {
                    'captions': captions,
                    'language': transcript.language_code,
                    'is_generated': transcript.is_generated
                }
        except Exception as e:
            st.error(f"Error getting captions: {e}")
            return None
        
        return None
    
    def detect_speakers_from_captions(self, captions: List[Dict]) -> Dict[str, List[str]]:
        """Detect potential speakers from caption timing and content patterns"""
        speakers = {'Speaker 1': [], 'Speaker 2': []}
        current_speaker = 'Speaker 1'
        
        # Simple speaker detection based on timing gaps and content patterns
        prev_end_time = 0
        
        for caption in captions:
            start_time = caption.get('start', 0)
            text = caption.get('text', '').strip()
            
            if not text:
                continue
            
            # If there's a significant gap (>3 seconds), might be speaker change
            time_gap = start_time - prev_end_time
            if time_gap > 3.0:
                # Switch speaker
                current_speaker = 'Speaker 2' if current_speaker == 'Speaker 1' else 'Speaker 1'
            
            # Look for dialogue indicators
            if any(indicator in text.lower() for indicator in ['said', 'asked', 'replied', '"', "'"]):
                # Might indicate dialogue - could be multiple speakers
                pass
            
            speakers[current_speaker].append(text)
            prev_end_time = caption.get('start', 0) + caption.get('duration', 0)
        
        # Filter out speakers with too little content
        return {k: v for k, v in speakers.items() if len(' '.join(v).split()) > 10}

def generate_voice_preview(text: str, voice_name: str, available_voices: Dict, is_cloned=False):
    """Generate and play voice preview with proper multi-speaker support and cloned voice support"""
    try:
        voice_config = available_voices[voice_name]
        model_name = voice_config['model']
        
        # Create longer, more natural preview text
        if len(text) < 100:
            # Extend short text to make a better preview
            longer_text = f"{text} This voice will be used throughout your entire audiobook to bring the story to life. The natural flow and intonation will make your content engaging and professional."
        else:
            longer_text = text[:500]  # Allow up to 500 characters for preview
        
        with st.spinner(f"Generating preview with {voice_name}..."):
            # Handle different voice types
            tts_kwargs = {'text': longer_text}
            
            if voice_config['type'] == 'cloned' and 'speaker_wav' in voice_config:
                # For cloned voices, ALWAYS use XTTS-v2 via API per small chunk
                st.info(f"🎬 Using XTTS-v2 for cloned voice: {voice_name}")
                
                # Check speaker file first
                st.write(f"📁 Speaker WAV file: {voice_config['speaker_wav']}")
                st.write(f"📄 File exists: {os.path.exists(voice_config['speaker_wav'])}")
                
                if not os.path.exists(voice_config['speaker_wav']):
                    st.error("❌ Speaker WAV file not found! Voice cloning cannot proceed.")
                    st.info("💡 Try cloning the voice again.")
                    return
                
                # Split preview text into small chunks (mirror long-form path)
                def _split_for_xtts_preview(text: str, max_chars: int = 200) -> List[str]:
                    import re
                    text = text.strip()
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    chunks = []
                    current = ""
                    for s in sentences:
                        if len(s) <= max_chars:
                            if len(current) + len(s) + 1 <= max_chars:
                                current = (current + " " + s).strip()
                            else:
                                if current:
                                    chunks.append(current)
                                current = s
                        else:
                            while len(s) > max_chars:
                                chunks.append(s[:max_chars])
                                s = s[max_chars:]
                            if s:
                                if current:
                                    chunks.append(current)
                                current = s
                    if current:
                        chunks.append(current)
                    return [c.strip() for c in chunks if c.strip()]
                
                # Read settings if available
                settings = st.session_state.get('xtts_settings', {}) if hasattr(st, 'session_state') else {}
                lang = (settings.get('language') or 'en') if isinstance(settings, dict) else 'en'
                max_chars = settings.get('chunk_size') if isinstance(settings, dict) else 200
                if not isinstance(max_chars, int):
                    max_chars = 200
                max_chars = max(80, min(600, max_chars))
                
                try:
                    # Use the bulletproof preview function
                    from xtts_preview_fix import generate_xtts_preview
                    
                    st.info("🔄 Generating preview with fixed XTTS...")
                    
                    # Generate preview audio
                    audio_bytes = generate_xtts_preview(
                        text=longer_text,
                        speaker_wav=voice_config['speaker_wav'],
                        language=lang or 'en',
                        chunk_size=max_chars
                    )
                    
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/wav', start_time=0)
                        st.success(f"✅ XTTS-v2 cloned voice preview generated! ({len(longer_text)} characters)")
                        st.success("🎉 This is your actual cloned voice!")
                    else:
                        st.error("❌ Preview generation failed. Check the console for details.")
                    return
                except ImportError:
                    st.error("❌ Missing dependencies for preview: pip install TTS pydub")
                    return
                except Exception as e:
                    st.error(f"❌ XTTS-v2 cloned voice preview failed: {e}")
                    st.info("💡 Try reducing chunk size or checking the speaker WAV file.")
                    return  # Exit without generating stock voice
            
            else:
                # For non-cloned voices, use standard TTS with safe loading
                try:
                    from TTS.api import TTS
                    
                    # Safe TTS initialization for regular voices
                    st.write(f"🔄 Loading {model_name}...")
                    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
                    
                    # Validate TTS model before use
                    if tts is None:
                        raise Exception("TTS initialization returned None")
                    if not hasattr(tts, 'model') or tts.model is None:
                        raise Exception("TTS model attribute is None or missing")
                    
                    st.write("✅ Standard TTS model loaded successfully")
                    
                    # Add speaker parameter for multi-speaker models
                    if voice_config['type'] == 'multi' and 'speaker' in voice_config:
                        tts_kwargs['speaker'] = voice_config['speaker']
                        st.write(f"🎤 Using speaker: {voice_config['speaker']}")
                    
                    # Generate audio with validation
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        st.write(f"🎯 Generating audio with {voice_name}...")
                        
                        # Final check before generation
                        if hasattr(tts, 'model') and tts.model is not None:
                            tts.tts_to_file(file_path=tmp_wav.name, **tts_kwargs)
                        else:
                            raise Exception("TTS model became None before generation")
                    
                    # Verify and play audio
                    if os.path.exists(tmp_wav.name) and os.path.getsize(tmp_wav.name) > 1024:
                        audio_bytes = open(tmp_wav.name, "rb").read()
                        st.audio(audio_bytes, format="audio/wav", start_time=0)
                        st.success(f"✅ Voice preview generated! ({len(longer_text)} characters)")
                        os.remove(tmp_wav.name)
                    else:
                        st.error("❌ Generated audio file is empty or invalid")
                        
                except Exception as e:
                    st.error(f"❌ Standard TTS generation failed: {e}")
                    st.info("💡 Try a different voice model or check your internet connection.")
            
    except Exception as e:
        st.error(f"❌ Error generating preview: {e}")
        st.info("💡 Try a different voice model or check your internet connection.")
        # Show debug info for troubleshooting
        if st.checkbox("Show debug info", key=f"debug_{voice_name}"):
            st.write(f"Voice config: {voice_config}")
            st.write(f"Error details: {str(e)}")

def load_cloned_voices() -> Dict:
    """Load cloned voices from YouTube voice cloning and integrate them"""
    cloned_voices = {}
    
    # Check for cloned voices in session state (from YouTube cloning)
    if 'cloned_voices' in st.session_state:
        for voice_name, voice_data in st.session_state['cloned_voices'].items():
            speaker_wav = voice_data.get('speaker_wav')
            if speaker_wav and os.path.exists(speaker_wav):
                cloned_voices[f"🎬 {voice_name} (Cloned)"] = {
                    "model": "tts_models/multilingual/multi-dataset/xtts_v2",  # Use XTTS-v2 for better cloning
                    "type": "cloned",
                    "speaker_wav": speaker_wav,
                    "description": f"🎆 XTTS-v2 cloned voice (1-min sample) from YouTube: {voice_data.get('url', 'Unknown source')}"
                }
    
    # Also check temp directory for any saved cloned models
    voice_models_dir = Path(tempfile.gettempdir()) / "audiobook_voice_models"
    if voice_models_dir.exists():
        try:
            # Look for voice model files
            for model_dir in voice_models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('voice_model_'):
                    model_name = model_dir.name.replace('voice_model_', '').replace('_', ' ').title()
                    
                    # Look for speaker wav files in voice_samples subdirectory
                    voice_samples_dir = model_dir / "voice_samples"
                    if voice_samples_dir.exists():
                        speaker_wav_files = list(voice_samples_dir.glob("*.wav"))
                        if speaker_wav_files:
                            # Use the first clip as the reference voice
                            best_clip = speaker_wav_files[0]
                            
                            # Try to find the longest clip for better voice cloning
                            if len(speaker_wav_files) > 1:
                                try:
                                    from pydub import AudioSegment
                                    longest_duration = 0
                                    for wav_file in speaker_wav_files:
                                        try:
                                            audio = AudioSegment.from_wav(wav_file)
                                            duration = len(audio) / 1000.0  # Convert to seconds
                                            if duration > longest_duration:
                                                longest_duration = duration
                                                best_clip = wav_file
                                        except:
                                            continue
                                except ImportError:
                                    pass  # pydub not available, use first clip
                            
                            cloned_voices[f"🎬 {model_name} (Custom)"] = {
                                "model": "tts_models/multilingual/multi-dataset/xtts_v2",
                                "type": "cloned",
                                "speaker_wav": str(best_clip),
                                "description": f"🎆 Custom voice model cloned from audio ({len(speaker_wav_files)} samples available)"
                            }
        except Exception as e:
            pass  # Silently fail if voice cloning module not available
    
    return cloned_voices

def clone_voice_from_youtube_integrated(youtube_url: str, speaker_name: str) -> bool:
    """Integrated YouTube voice cloning that adds directly to available voices"""
    try:
        # This would integrate with your voice cloning module
        from voice_cloning import create_voice_from_youtube_url
        
        # Create temporary directory for this cloning session
        temp_clone_dir = Path(tempfile.mkdtemp()) / "youtube_clone"
        temp_clone_dir.mkdir(exist_ok=True)
        
        with st.spinner(f"🎬 Cloning voice for {speaker_name}..."):
            success = create_voice_from_youtube_url(
                url=youtube_url,
                speaker_name=speaker_name,
                output_dir=str(temp_clone_dir)
            )
        
        if success:
            # Look for the combined speaker_wav.wav file first (optimized for XTTS-v2)
            speaker_wav_file = None
            voice_model_dirs = list(temp_clone_dir.glob("voice_model_*"))
            
            if voice_model_dirs:
                # Check for the combined speaker sample
                combined_wav = voice_model_dirs[0] / "speaker_wav.wav"
                if combined_wav.exists():
                    speaker_wav_file = combined_wav
                    st.success(f"✅ Found optimized 1-minute speaker sample for XTTS-v2")
                else:
                    # Fall back to individual clips
                    wav_files = list(voice_model_dirs[0].rglob("*.wav"))
                    if wav_files:
                        speaker_wav_file = wav_files[0]
                        st.warning("⚠️ Using individual clip - may not work as well with XTTS-v2")
            
            if speaker_wav_file:
                # Store in session state for immediate use
                if 'cloned_voices' not in st.session_state:
                    st.session_state['cloned_voices'] = {}
                
                st.session_state['cloned_voices'][speaker_name] = {
                    'speaker_wav': str(speaker_wav_file),
                    'source': 'youtube',
                    'url': youtube_url
                }
                
                # Analyze the audio duration
                try:
                    import librosa
                    y, sr = librosa.load(str(speaker_wav_file), sr=None)
                    duration = len(y) / sr
                    if duration >= 45.0:
                        st.success(f"✅ Speaker sample duration: {duration:.1f}s (excellent for XTTS-v2)")
                    elif duration >= 30.0:
                        st.info(f"📏 Speaker sample duration: {duration:.1f}s (good for XTTS-v2)")
                    else:
                        st.warning(f"⚠️ Speaker sample duration: {duration:.1f}s (may be too short for optimal XTTS-v2 results)")
                except:
                    st.info("Speaker sample ready for XTTS-v2")
                
                return True
        
        return False
        
    except ImportError:
        st.error("❌ Voice cloning requires additional packages: `pip install yt-dlp pydub`")
        return False
    except Exception as e:
        st.error(f"❌ Voice cloning error: {e}")
        return False

def clone_voice_from_audio_file(audio_file_path: Path, speaker_name: str) -> bool:
    """Clone voice from uploaded MP3/audio file"""
    try:
        import tempfile
        from pathlib import Path
        import shutil
        
        # Validate input
        audio_file_path = Path(audio_file_path)
        if not audio_file_path.exists():
            st.error(f"Audio file not found: {audio_file_path}")
            return False
        
        # Create temporary directory for processing
        temp_clone_dir = Path(tempfile.mkdtemp()) / "mp3_clone"
        temp_clone_dir.mkdir(exist_ok=True)
        
        st.info(f"🎵 Processing audio file: {audio_file_path.name}")
        
        # Process the audio file for voice cloning
        with st.spinner(f"🎵 Processing audio for voice cloning..."):
            # Check file format and duration
            try:
                from pydub import AudioSegment
                import librosa
                
                # Load and analyze audio
                audio = AudioSegment.from_file(audio_file_path)
                duration = len(audio) / 1000.0  # Convert to seconds
                
                st.info(f"📏 Audio duration: {duration:.1f} seconds")
                
                if duration < 3.0:
                    st.warning("⚠️ Audio is quite short (less than 3 seconds). Voice cloning quality may be reduced.")
                elif duration > 60.0:
                    st.warning("⚠️ Audio is longer than 60 seconds. Using first 60 seconds for voice cloning.")
                    audio = audio[:60000]  # Trim to first 60 seconds
                    duration = 60.0
                
                # Convert to format suitable for XTTS-v2
                # Convert to mono if stereo
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Set sample rate to 22050 (standard for voice cloning)
                audio = audio.set_frame_rate(22050)
                
                # Normalize audio
                audio = audio.normalize()
                
                # Save processed audio
                processed_audio_path = temp_clone_dir / "speaker_wav.wav"
                audio.export(processed_audio_path, format="wav")
                
                st.success(f"✅ Audio processed: {duration:.1f}s, 22kHz mono")
                
            except ImportError:
                st.error("❌ Required audio processing packages not found. Install with: pip install pydub librosa")
                return False
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                return False
        
        # Store in session state for immediate use
        if 'cloned_voices' not in st.session_state:
            st.session_state['cloned_voices'] = {}
        
        st.session_state['cloned_voices'][speaker_name] = {
            'speaker_wav': str(processed_audio_path),
            'source': 'mp3_upload',
            'original_file': str(audio_file_path),
            'duration': duration
        }
        
        # Quality feedback
        if duration >= 10.0:
            st.success(f"✅ High quality voice sample ({duration:.1f}s) - excellent for XTTS-v2")
        elif duration >= 5.0:
            st.info(f"📏 Good quality voice sample ({duration:.1f}s) - suitable for XTTS-v2")
        else:
            st.warning(f"⚠️ Short voice sample ({duration:.1f}s) - may need longer audio for best results")
        
        return True
        
    except ImportError:
        st.error("❌ Audio processing requires additional packages: `pip install pydub librosa`")
        return False
    except Exception as e:
        st.error(f"❌ Voice cloning from audio file error: {e}")
        return False

def voice_editor_ui():
    st.title("🎤 Voice Editor & Speaker Assignment")
    
    # Load better, more natural voice models with cloned voices integration
    available_voices = {
        # High-quality neural voices with better naturalness
        "Neural Female (Clear & Natural)": {
            "model": "tts_models/en/ljspeech/vits",
            "type": "single",
            "description": "🎆 High-quality neural female voice with natural prosody. Best for audiobooks."
        },
        "Neural Female (Expressive)": {
            "model": "tts_models/en/ljspeech/vits--neon", 
            "type": "single",
            "description": "🎭 Expressive female voice with emotional range. Great for storytelling."
        },
        "Multi-Speaker Female (Jenny)": {
            "model": "tts_models/en/vctk/vits",
            "type": "multi",
            "speaker": "p225",
            "description": "🎤 Clear female voice with natural cadence. Excellent for long-form content."
        },
        "Multi-Speaker Male (David)": {
            "model": "tts_models/en/vctk/vits",
            "type": "multi",
            "speaker": "p226",
            "description": "🎤 Professional male voice with warm tone. Perfect for narratives."
        },
        "Multi-Speaker Female (Sarah)": {
            "model": "tts_models/en/vctk/vits",
            "type": "multi",
            "speaker": "p228",
            "description": "🎤 Engaging female voice with clear diction. Great for educational content."
        },
        "Multi-Speaker Male (James)": {
            "model": "tts_models/en/vctk/vits",
            "type": "multi", 
            "speaker": "p227",
            "description": "🎤 Distinguished male voice with authoritative tone. Ideal for non-fiction."
        }
    }
    
    # Check for YouTube cloned voices and add them to available voices
    cloned_voices = load_cloned_voices()
    if cloned_voices:
        st.info(f"🎆 Found {len(cloned_voices)} custom cloned voices!")
        available_voices.update(cloned_voices)
    
    # Check if we have detected speakers from PDF processing
    detected_speakers = st.session_state.get('detected_speakers', {})
    has_speakers = bool(detected_speakers)
    
    # XTTS Settings (applies to xtts_v2 and cloned voices)
    with st.expander("⚙️ XTTS Voice Settings & Expressiveness (for xtts_v2 and cloned voices)", expanded=False):
        # Voice Expressiveness Presets
        st.subheader("🎭 Voice Expressiveness Presets")
        
        presets = {
            "📖 Neutral Narration": {
                "description": "Balanced, clear narration for general content",
                "temperature": 0.65,
                "top_k": 50,
                "top_p": 0.85,
                "speed": 1.0,
                "repetition_penalty": 2.0,
                "length_penalty": 1.0,
                "emotion": "neutral"
            },
            "🎬 Dramatic & Expressive": {
                "description": "Dynamic, emotional delivery for fiction and storytelling",
                "temperature": 0.85,
                "top_k": 70,
                "top_p": 0.92,
                "speed": 0.95,
                "repetition_penalty": 1.8,
                "length_penalty": 1.1,
                "emotion": "expressive"
            },
            "😊 Warm & Friendly": {
                "description": "Cheerful, engaging tone for children's books or light content",
                "temperature": 0.75,
                "top_k": 60,
                "top_p": 0.88,
                "speed": 1.05,
                "repetition_penalty": 1.9,
                "length_penalty": 1.0,
                "emotion": "happy"
            },
            "🎓 Authoritative & Clear": {
                "description": "Professional, confident delivery for non-fiction and educational content",
                "temperature": 0.55,
                "top_k": 40,
                "top_p": 0.80,
                "speed": 0.95,
                "repetition_penalty": 2.2,
                "length_penalty": 1.0,
                "emotion": "serious"
            },
            "🌙 Calm & Soothing": {
                "description": "Gentle, relaxing voice for meditation or bedtime stories",
                "temperature": 0.60,
                "top_k": 45,
                "top_p": 0.82,
                "speed": 0.90,
                "repetition_penalty": 2.1,
                "length_penalty": 1.05,
                "emotion": "calm"
            },
            "⚡ Energetic & Upbeat": {
                "description": "Fast-paced, enthusiastic delivery for action or motivational content",
                "temperature": 0.80,
                "top_k": 65,
                "top_p": 0.90,
                "speed": 1.10,
                "repetition_penalty": 1.7,
                "length_penalty": 0.95,
                "emotion": "excited"
            },
            "🕵️ Mysterious & Suspenseful": {
                "description": "Intriguing, tension-building tone for thrillers and mysteries",
                "temperature": 0.70,
                "top_k": 55,
                "top_p": 0.86,
                "speed": 0.92,
                "repetition_penalty": 2.0,
                "length_penalty": 1.08,
                "emotion": "mysterious"
            },
            "🤖 Robotic & Consistent": {
                "description": "Ultra-consistent, minimal variation for technical documentation",
                "temperature": 0.40,
                "top_k": 30,
                "top_p": 0.75,
                "speed": 1.0,
                "repetition_penalty": 2.5,
                "length_penalty": 1.0,
                "emotion": "monotone"
            },
            "🎯 Custom Settings": {
                "description": "Manually configure all parameters for fine control",
                "temperature": None,  # Will use current/default values
                "top_k": None,
                "top_p": None,
                "speed": None,
                "repetition_penalty": None,
                "length_penalty": None,
                "emotion": "custom"
            }
        }
        
        # Preset selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_preset = st.selectbox(
                "Choose a voice style preset:",
                list(presets.keys()),
                index=0,
                help="Select a preset to automatically configure voice expressiveness"
            )
        
        with col2:
            # Emotion intensity slider (affects how strongly the preset is applied)
            emotion_intensity = st.slider(
                "Intensity",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1,
                help="Adjust how strongly the preset affects the voice (1.0 = normal)"
            )
        
        # Display preset description
        preset_config = presets[selected_preset]
        st.info(f"**{selected_preset}**: {preset_config['description']}")
        
        # Apply preset or use custom settings
        if selected_preset != "🎯 Custom Settings":
            # Apply preset with intensity scaling
            preset_temp = preset_config["temperature"]
            preset_top_k = preset_config["top_k"]
            preset_top_p = preset_config["top_p"]
            preset_speed = preset_config["speed"]
            preset_rep_pen = preset_config["repetition_penalty"]
            preset_len_pen = preset_config["length_penalty"]
            
            # Scale some parameters based on intensity
            if emotion_intensity != 1.0:
                # Temperature scales with intensity (more intensity = more variation)
                preset_temp = min(1.0, preset_temp * (0.7 + 0.3 * emotion_intensity))
                # Speed scales slightly with intensity
                preset_speed = preset_speed * (0.9 + 0.1 * emotion_intensity)
                # Top-k scales with intensity (more intensity = more vocabulary)
                preset_top_k = int(preset_top_k * (0.8 + 0.2 * emotion_intensity))
            
            st.success(f"✨ Using {selected_preset} preset with {emotion_intensity:.1f}x intensity")
            
            # Show the preset values that will be used
            with st.expander("View preset parameters", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{preset_temp:.2f}")
                    st.metric("Top-K", preset_top_k)
                with col2:
                    st.metric("Top-P", f"{preset_top_p:.2f}")
                    st.metric("Speed", f"{preset_speed:.2f}")
                with col3:
                    st.metric("Rep. Penalty", f"{preset_rep_pen:.1f}")
                    st.metric("Len. Penalty", f"{preset_len_pen:.2f}")
            
            # Store preset values
            use_temp = preset_temp
            use_top_k = preset_top_k
            use_top_p = preset_top_p
            use_speed = preset_speed
            use_rep_pen = preset_rep_pen
            use_len_pen = preset_len_pen
            use_emotion = preset_config["emotion"]
        else:
            # Custom settings - show full controls
            st.info("🎛️ **Manual Control Mode** - Fine-tune all parameters below")
            
            # Load current settings or defaults
            current_settings = st.session_state.get('xtts_settings', {}) if hasattr(st, 'session_state') else {}
            default_chunk = int(current_settings.get('chunk_size', 250) or 250)
            default_lang = str(current_settings.get('language', 'en') or 'en')
            default_temp = float(current_settings.get('temperature', 0.65) or 0.65)
            default_top_k = int(current_settings.get('top_k', 50) or 50)
            default_top_p = float(current_settings.get('top_p', 0.85) or 0.85)
            default_len_pen = float(current_settings.get('length_penalty', 1.0) or 1.0)
            default_rep_pen = float(current_settings.get('repetition_penalty', 2.0) or 2.0)
            default_speed = float(current_settings.get('speed', 1.0) or 1.0)
            
            use_temp = default_temp
            use_top_k = default_top_k
            use_top_p = default_top_p
            use_speed = default_speed
            use_rep_pen = default_rep_pen
            use_len_pen = default_len_pen
            use_emotion = "custom"
        
        # Advanced settings section
        st.divider()
        advanced_header = "🔧 Advanced Settings" if selected_preset == "🎯 Custom Settings" else "🔧 Fine-tune Settings"
        with st.expander(advanced_header, expanded=(selected_preset == "🎯 Custom Settings")):
            st.write("**Processing Settings:**")
            chunk_size = st.slider(
                "Chunk size (characters)",
                min_value=100,
                max_value=400,
                value=max(100, min(400, default_chunk)),
                help="250 characters is optimal - respects sentence boundaries while maintaining consistency"
            )
            language = st.selectbox(
                "Language",
                options=["en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "fi", "tr", "pl", "ru", "uk", "ar", "hi", "ja", "ko", "zh"],
                index=["en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "fi", "tr", "pl", "ru", "uk", "ar", "hi", "ja", "ko", "zh"].index(default_lang) if default_lang in ["en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "fi", "tr", "pl", "ru", "uk", "ar", "hi", "ja", "ko", "zh"] else 0,
                help="Language code to pass to XTTS"
            )
            st.write("**Voice Parameters:**")
            cols = st.columns(2)
            with cols[0]:
                if selected_preset == "🎯 Custom Settings":
                    temperature = st.slider(
                        "🌡️ Temperature", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=float(max(0.1, min(1.0, use_temp))), 
                        step=0.05, 
                        help="Controls randomness: Lower (0.4-0.6) = consistent, Higher (0.7-0.9) = expressive"
                    )
                    top_k = st.slider(
                        "🎯 Top-K", 
                        min_value=0, 
                        max_value=100, 
                        value=int(max(0, min(100, use_top_k))), 
                        step=5, 
                        help="Vocabulary diversity: Lower (30-40) = focused, Higher (60-80) = varied"
                    )
                    speed = st.slider(
                        "⏱️ Speaking speed", 
                        min_value=0.8, 
                        max_value=1.2, 
                        value=float(max(0.8, min(1.2, use_speed))), 
                        step=0.05, 
                        help="Narration pace: 0.9 = slow, 1.0 = normal, 1.1 = fast"
                    )
                else:
                    # Show preset values as disabled/read-only
                    st.metric("🌡️ Temperature", f"{use_temp:.2f}")
                    st.metric("🎯 Top-K", use_top_k)
                    st.metric("⏱️ Speed", f"{use_speed:.2f}")
                    # Allow override
                    temperature = use_temp
                    top_k = use_top_k
                    speed = use_speed
                    
            with cols[1]:
                if selected_preset == "🎯 Custom Settings":
                    top_p = st.slider(
                        "☂️ Top-P (nucleus)", 
                        min_value=0.5, 
                        max_value=0.95, 
                        value=float(max(0.5, min(0.95, use_top_p))), 
                        step=0.05, 
                        help="Token selection: Lower (0.7-0.8) = focused, Higher (0.9-0.95) = creative"
                    )
                    length_penalty = st.slider(
                        "📏 Length penalty", 
                        min_value=0.8, 
                        max_value=1.2, 
                        value=float(max(0.8, min(1.2, use_len_pen))), 
                        step=0.05, 
                        help="Sentence length: <1.0 = shorter, 1.0 = natural, >1.0 = longer"
                    )
                    repetition_penalty = st.slider(
                        "🔁 Repetition penalty", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=float(max(1.0, min(3.0, use_rep_pen))), 
                        step=0.1, 
                        help="Prevents loops: 1.5-2.0 = balanced, 2.0-2.5 = strict"
                    )
                else:
                    # Show preset values
                    st.metric("☂️ Top-P", f"{use_top_p:.2f}")
                    st.metric("📏 Length Penalty", f"{use_len_pen:.2f}")
                    st.metric("🔁 Rep. Penalty", f"{use_rep_pen:.1f}")
                    # Use preset values
                    top_p = use_top_p
                    length_penalty = use_len_pen
                    repetition_penalty = use_rep_pen
        
        # Store settings with preset info
        st.session_state['xtts_settings'] = {
            "chunk_size": int(chunk_size),
            "language": language,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "length_penalty": float(length_penalty),
            "repetition_penalty": float(repetition_penalty),
            "speed": float(speed),
            "preset": selected_preset if 'selected_preset' in locals() else "📖 Neutral Narration",
            "emotion": use_emotion if 'use_emotion' in locals() else "neutral",
            "intensity": emotion_intensity if 'emotion_intensity' in locals() else 1.0
        }
        # Display current configuration summary
        st.divider()
        if 'selected_preset' in locals() and selected_preset != "🎯 Custom Settings":
            config_emoji = selected_preset.split()[0]
            st.success(f"{config_emoji} **Active Style**: {selected_preset} at {emotion_intensity:.1f}x intensity")
            
            # Style-specific tips
            style_tips = {
                "📖 Neutral Narration": "Perfect for general audiobooks and factual content",
                "🎬 Dramatic & Expressive": "Great for fiction with emotional scenes and dialogue",
                "😊 Warm & Friendly": "Ideal for children's books and uplifting content",
                "🎓 Authoritative & Clear": "Best for textbooks, documentaries, and instructional material",
                "🌙 Calm & Soothing": "Wonderful for meditation guides and bedtime stories",
                "⚡ Energetic & Upbeat": "Excellent for action scenes and motivational speeches",
                "🕵️ Mysterious & Suspenseful": "Perfect for mystery novels and thriller narratives",
                "🤖 Robotic & Consistent": "Optimal for technical manuals and consistent delivery"
            }
            if selected_preset in style_tips:
                st.info(f"💡 {style_tips[selected_preset]}")
        else:
            # Custom mode warnings
            if temperature > 0.7:
                st.warning("⚠️ High temperature may cause voice inconsistency. Consider lowering to 0.5-0.7.")
            if repetition_penalty < 1.5:
                st.warning("⚠️ Low repetition penalty may cause voice loops. Consider raising to 1.5-2.5.")
            if chunk_size < 200:
                st.info("💡 Smaller chunks may cut sentences. Consider 200-300 characters.")
            
            st.info(f"🎛️ **Custom Mode**: temp: {temperature:.2f}, top_k: {top_k}, top_p: {top_p:.2f}, speed: {speed:.2f}")
        
        # Quick preview with these settings (XTTS-only voices)
        xtts_voice_names = [name for name, cfg in available_voices.items() if (cfg.get('type') == 'cloned') or ('xtts' in str(cfg.get('model', '')).lower())]
        if xtts_voice_names:
            st.write("Preview with current settings:")
            pv_voice = st.selectbox("Voice", xtts_voice_names, key="xtts_settings_preview_voice")
            pv_text = st.text_area("Text", value="Hello! This is a preview using current XTTS settings.", height=80, key="xtts_settings_preview_text")
            if st.button("🎧 Preview (XTTS)", key="xtts_settings_preview_btn"):
                generate_voice_preview(pv_text, pv_voice, available_voices)
        else:
            st.info("No XTTS/cloned voices available to preview yet.")
    
    if has_speakers:
        st.success(f"📚 Found {len(detected_speakers)} speakers in your audiobook!")
        
        # Automatic speaker-to-voice mapping
        st.subheader("🎭 Speaker Voice Assignment")
        st.write("Assign voices to each detected speaker in your audiobook:")
        
        # Initialize speaker voice mapping if not exists
        if 'speaker_voice_mapping' not in st.session_state:
            st.session_state.speaker_voice_mapping = {}
            
            # Auto-assign voices to speakers
            voice_list = list(available_voices.keys())
            for i, speaker_name in enumerate(detected_speakers.keys()):
                # Assign different voices, cycling through available ones
                assigned_voice = voice_list[i % len(voice_list)]
                st.session_state.speaker_voice_mapping[speaker_name] = assigned_voice
        
        # Display speaker assignments
        for speaker_name, speaker_data in detected_speakers.items():
            with st.expander(f"🎤 {speaker_name} ({speaker_data.dialogue_count} dialogues, {speaker_data.word_count} words)", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Speaker Details:**")
                    st.write(f"• Dialogues: {speaker_data.dialogue_count}")
                    st.write(f"• Words: {speaker_data.word_count}")
                    
                    # Show characteristics if available
                    if hasattr(speaker_data, 'characteristics'):
                        chars = speaker_data.characteristics
                        if 'formality_score' in chars:
                            formality = chars['formality_score']
                            st.write(f"• Formality: {formality:.2f} {'(Formal)' if formality > 0.6 else '(Casual)'}")
                        if 'avg_sentence_length' in chars:
                            st.write(f"• Avg sentence: {chars['avg_sentence_length']} words")
                    
                    # Show sample dialogue
                    if hasattr(speaker_data, 'characteristics') and 'dialogue_snippets' in speaker_data.characteristics:
                        st.write("**Sample dialogue:**")
                        for snippet in speaker_data.characteristics['dialogue_snippets'][:2]:
                            st.write(f"_{snippet[:80]}..._")
                
                with col2:
                    st.write("**Voice Assignment:**")
                    
                    # Voice selection
                    current_voice = st.session_state.speaker_voice_mapping.get(speaker_name, list(available_voices.keys())[0])
                    selected_voice = st.selectbox(
                        f"Voice for {speaker_name}",
                        list(available_voices.keys()),
                        index=list(available_voices.keys()).index(current_voice) if current_voice in available_voices else 0,
                        key=f"voice_select_{speaker_name}"
                    )
                    
                    # Update mapping
                    st.session_state.speaker_voice_mapping[speaker_name] = selected_voice
                    
                    # Show voice description
                    st.info(available_voices[selected_voice]['description'])
                    
                    # Voice preview with actual speaker dialogue
                    if st.button(f"🎵 Preview {speaker_name}'s Voice", key=f"preview_{speaker_name}"):
                        preview_text = "Hello, this is how I sound in your audiobook."
                        if hasattr(speaker_data, 'characteristics') and 'dialogue_snippets' in speaker_data.characteristics:
                            preview_text = speaker_data.characteristics['dialogue_snippets'][0][:150]
                        
                        generate_voice_preview(preview_text, selected_voice, available_voices)
        
        # Save mapping button
        if st.button("💾 Save Voice Assignments"):
            st.success("✅ Voice assignments saved! Ready for audiobook generation.")
            
        st.divider()
    
    # Voice testing section
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("🎵 **Test Any Voice**")
        
        # Voice selection for testing
        test_voice = st.selectbox(
            "Choose a voice to test",
            list(available_voices.keys()),
            key="test_voice_select"
        )
        
        # Custom text input
        test_text = st.text_area(
            "Enter text to test", 
            value="Hello! This is a voice test. How do I sound for your audiobook?",
            height=100,
            key="test_text"
        )
        
        if st.button("🎵 Test Voice", key="test_voice_btn"):
            generate_voice_preview(test_text, test_voice, available_voices)
    
    with col2:
        st.write("📺 **YouTube Voice Analysis**")
        
        if not YOUTUBE_AVAILABLE:
            st.error("❌ YouTube integration requires additional packages.")
            st.code("pip install yt-dlp youtube-transcript-api")
        else:
            youtube_url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Analyze YouTube video captions for voice patterns",
                key="youtube_url"
            )
            
            if youtube_url and st.button("🔍 Analyze Video", key="analyze_youtube"):
                with st.spinner("Extracting captions..."):
                    extractor = YouTubeCaptionExtractor()
                    caption_data = extractor.get_captions(youtube_url)
                    
                    if caption_data:
                        captions = caption_data['captions']
                        st.success(f"✅ Extracted {len(captions)} caption segments")
                        
                        # Store captions for voice cloning integration
                        st.session_state['youtube_captions'] = captions
                        st.session_state['youtube_caption_text'] = ' '.join([c.get('text', '') for c in captions])
                        
                        # Show sample
                        sample_text = ' '.join([c.get('text', '') for c in captions[:3]])
                        st.write(f"**Sample text:** {sample_text[:200]}...")
                        
                        # Test voice with YouTube content
                        if st.button("🎵 Test Voice with YouTube Content", key="test_youtube_voice"):
                            youtube_sample = sample_text[:150]
                            generate_voice_preview(youtube_sample, test_voice, available_voices)
                    else:
                        st.error("❌ Could not extract captions from this video.")
    
    # Voice cloning integration section
    if not has_speakers:
        st.info("💡 **Tip:** Process a PDF with speakers in the first tab to see automatic voice assignments here.")
    
    # Show cloned voices if any exist
    st.subheader("🎬 Voice Cloning Integration")
    
    # Add direct voice cloning interface
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🎥 Clone a Voice from YouTube", expanded=False):
            if YOUTUBE_AVAILABLE:
                clone_url = st.text_input(
                    "YouTube URL for voice cloning",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="Provide a YouTube video with clear speech for voice cloning",
                    key="clone_youtube_url"
                )
                
                clone_name = st.text_input(
                    "Voice name",
                    placeholder="Speaker Name",
                    key="clone_voice_name"
                )
                
                if st.button("🎬 Clone Voice from YouTube", key="clone_voice_btn"):
                    if clone_url and clone_name:
                        if clone_voice_from_youtube_integrated(clone_url, clone_name):
                            st.success(f"✅ Successfully cloned voice '{clone_name}'! It should now appear in the voice list above.")
                            st.experimental_rerun()  # Refresh to show the new voice
                        else:
                            st.error("❌ Voice cloning failed. Please check the URL and try again.")
                    else:
                        st.warning("⚠️ Please provide both YouTube URL and voice name.")
            else:
                st.error("❌ YouTube integration requires: `pip install yt-dlp youtube-transcript-api`")
    
    with col2:
        with st.expander("🎵 Clone a Voice from MP3/Audio File", expanded=False):
            st.write("Upload an audio file (MP3, WAV, etc.) to clone a voice:")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'ogg', 'flac', 'm4a'],
                help="Upload a clear audio file with speech (5-60 seconds recommended)",
                key="upload_audio_file"
            )
            
            # Voice name input
            mp3_clone_name = st.text_input(
                "Voice name for MP3 clone",
                placeholder="Custom Voice Name",
                key="mp3_clone_voice_name"
            )
            
            # Clone button
            if st.button("🎵 Clone Voice from Audio", key="clone_mp3_btn"):
                if uploaded_file and mp3_clone_name:
                    # Save uploaded file to temporary location
                    temp_audio_path = Path(tempfile.mkdtemp()) / uploaded_file.name
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Clone voice from the uploaded file
                    if clone_voice_from_audio_file(temp_audio_path, mp3_clone_name):
                        st.success(f"✅ Successfully cloned voice '{mp3_clone_name}' from {uploaded_file.name}!")
                        st.info("🎉 Your cloned voice should now appear in the voice list above.")
                        st.experimental_rerun()  # Refresh to show the new voice
                    else:
                        st.error("❌ Voice cloning from audio file failed. Please try a different file.")
                    
                    # Clean up temp file
                    try:
                        temp_audio_path.unlink()
                    except:
                        pass
                        
                elif not uploaded_file:
                    st.warning("⚠️ Please upload an audio file first.")
                elif not mp3_clone_name:
                    st.warning("⚠️ Please provide a name for the cloned voice.")
            
            # Tips for better results
            st.info("💡 **Tips for best results:**")
            st.write("• Use 10-60 seconds of clear speech")
            st.write("• Avoid background music/noise")
            st.write("• Single speaker only")
            st.write("• Good audio quality (no compression artifacts)")
    
    # Check for cloned voice models in temp directory (not in app folder)
    voice_models_dir = Path(tempfile.gettempdir()) / "audiobook_voice_models"
    if voice_models_dir.exists():
        try:
            from voice_cloning import VoiceCloner
            cloner = VoiceCloner()
            cloned_models = cloner.list_available_voices(voice_models_dir)
            
            if cloned_models:
                st.write(f"🎭 Found {len(cloned_models)} cloned voice models:")
                
                for model in cloned_models:
                    with st.expander(f"🎤 {model['name']} (Quality: {model.get('quality', 0):.2f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Clips:** {model.get('clips', 0)}")
                            st.write(f"**Duration:** {model.get('duration', 0):.1f}s")
                            st.write(f"**Path:** `{model['path']}`")
                        
                        with col2:
                            st.write("**Integration Options:**")
                            if st.button(f"Use {model['name']} for Speaker", key=f"use_cloned_{model['name']}"):
                                st.info("🛠️ Cloned voice integration will be available in the next version.")
                                # TODO: Integrate cloned voices with TTS system
                            
                            if st.button(f"Test {model['name']}", key=f"test_cloned_{model['name']}"):
                                st.info("🛠️ Direct testing of cloned voices will be available in the next version.")
                                # TODO: Add cloned voice testing
            else:
                st.info("💡 No cloned voices found. Use the Voice Cloning section above to create custom voices.")
                
        except ImportError:
            st.info("💡 Voice cloning integration requires the voice_cloning module.")
    else:
        st.info("💡 No voice models directory found. Clone a voice first to see integration options.")
