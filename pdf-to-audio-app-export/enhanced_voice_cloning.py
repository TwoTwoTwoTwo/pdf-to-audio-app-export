#!/usr/bin/env python3
"""
Enhanced Voice Cloning with XTTS-v2 Support
High-quality voice cloning for multiple speakers and languages
"""

# CRITICAL: Install XTTS import hook BEFORE any TTS/Transformers imports
try:
    import import_hook_xtts_patch  # installs hook on import
except Exception:
    pass

# CRITICAL: Apply XTTS-v2 fixes BEFORE any other TTS/Transformers imports
try:
    from xtts_v2_fix import apply_xtts_v2_transformers_fix, suppress_transformers_warnings
    suppress_transformers_warnings()
    apply_xtts_v2_transformers_fix()
except ImportError:
    # If fix module isn't available, we'll handle errors gracefully
    pass

import os
import re
import json
import torch
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import shutil
from datetime import datetime
import logging

# Suppress torchaudio backend dispatch warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*par-call.*backend.*dispatch.*")
    warnings.filterwarnings("ignore", message=".*backend.*implementation.*directly.*")
    warnings.filterwarnings("ignore", message=".*I/O functions.*backend.*dispatch.*")
    import torchaudio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XTTSVoiceCloner:
    """Enhanced voice cloning using XTTS-v2"""
    
    def __init__(self, device: Optional[str] = None, output_dir: Optional[Path] = None):
        """Initialize the XTTS voice cloner"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir or Path("./voice_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # XTTS-v2 model
        self.xtts_model = None
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", 
            "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
        ]
        
        # Voice quality thresholds
        self.min_audio_duration = 3.0  # seconds
        self.max_audio_duration = 30.0  # seconds per clip
        self.target_sample_rate = 22050
        
    def load_xtts_model(self, progress_callback=None) -> bool:
        """Load the XTTS-v2 model"""
        try:
            if progress_callback:
                progress_callback("🔄 Loading XTTS-v2 model...")
            
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            
            if progress_callback:
                progress_callback("✅ XTTS-v2 model loaded successfully")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Failed to load XTTS-v2: {str(e)}")
            return False
    
    def clone_voice_from_audio(
        self,
        audio_file: Path,
        speaker_name: str,
        language: str = "en",
        progress_callback=None
    ) -> Optional[Dict[str, Any]]:
        """Clone a voice from a single audio file"""
        
        if not self.xtts_model:
            if not self.load_xtts_model(progress_callback):
                return None
        
        try:
            if progress_callback:
                progress_callback(f"🎤 Cloning voice for {speaker_name}...")
            
            # Validate language
            if language not in self.supported_languages:
                if progress_callback:
                    progress_callback(f"⚠️ Language {language} not supported, using 'en'")
                language = "en"
            
            # Process the audio file
            processed_audio = self._process_audio_for_cloning(audio_file, progress_callback)
            if not processed_audio:
                return None
            
            # Create voice model directory
            model_dir = self.output_dir / f"xtts_voice_{speaker_name.lower().replace(' ', '_')}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the processed reference audio
            reference_path = model_dir / "reference.wav"
            processed_audio.export(reference_path, format="wav")
            
            # Test the voice with a sample
            test_text = "Hello, this is a test of the voice cloning. How does this sound?"
            test_output = model_dir / "test_sample.wav"
            
            if progress_callback:
                progress_callback("🧪 Testing cloned voice...")
            
            try:
                self.xtts_model.tts_to_file(
                    text=test_text,
                    file_path=str(test_output),
                    speaker_wav=str(reference_path),
                    language=language
                )
                
                # Analyze quality
                quality_score = self._analyze_voice_quality(test_output, reference_path)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"⚠️ Voice test failed: {e}")
                quality_score = 0.5
            
            # Create voice metadata
            voice_data = {
                "name": speaker_name,
                "language": language,
                "created": datetime.now().isoformat(),
                "reference_audio": str(reference_path.relative_to(self.output_dir)),
                "test_audio": str(test_output.relative_to(self.output_dir)) if test_output.exists() else None,
                "quality_score": quality_score,
                "duration": len(processed_audio) / 1000.0,  # in seconds
                "sample_rate": processed_audio.frame_rate,
                "model": "xtts_v2",
                "supported_languages": self.supported_languages,
                "device": self.device
            }
            
            # Save metadata
            metadata_path = model_dir / "voice_config.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(voice_data, f, indent=2, ensure_ascii=False)
            
            if progress_callback:
                progress_callback(f"✅ Voice cloned successfully! Quality: {quality_score:.2f}")
            
            return voice_data
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Voice cloning failed: {str(e)}")
            return None
    
    def clone_voice_from_youtube(
        self,
        youtube_url: str,
        speaker_name: str,
        language: str = "en",
        max_clips: int = 5,
        progress_callback=None
    ) -> Optional[Dict[str, Any]]:
        """Clone voice from YouTube video using robust extraction methods"""
        
        if progress_callback:
            progress_callback(f"📺 Extracting audio from YouTube...")
        
        # Use the robust YouTube audio extractor
        try:
            from youtube_audio_extractor import extract_youtube_audio
        except ImportError:
            if progress_callback:
                progress_callback("❌ YouTube extractor not available, using fallback method")
            return self._fallback_youtube_extraction(youtube_url, speaker_name, language, max_clips, progress_callback)
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "extracted_audio.wav"
            
            # Extract audio using robust methods
            success = extract_youtube_audio(youtube_url, audio_file, progress_callback)
            
            if not success:
                if progress_callback:
                    progress_callback("❌ Failed to extract audio from YouTube")
                return None
            
            # Validate the extracted file
            file_size_mb = audio_file.stat().st_size / 1024 / 1024
            if file_size_mb < 0.5:  # Less than 500KB
                if progress_callback:
                    progress_callback(f"❌ Extracted file too small ({file_size_mb:.1f}MB) - likely empty")
                return None
            
            if progress_callback:
                progress_callback(f"✅ Successfully extracted {file_size_mb:.1f}MB of audio")
            
            # Now process the extracted audio for voice cloning
            
            # Process the extracted audio for voice cloning
            if progress_callback:
                progress_callback("🎵 Processing extracted audio...")
            
            # Extract good quality clips for voice cloning
            clips = self._extract_voice_clips(audio_file, max_clips, progress_callback)
            
            if not clips:
                if progress_callback:
                    progress_callback("❌ No suitable voice clips found")
                return None
            
            # Use the best clip for cloning
            best_clip = self._select_best_clip(clips)
            
            if not best_clip:
                if progress_callback:
                    progress_callback("❌ Could not select best clip for cloning")
                return None
            
            return self.clone_voice_from_audio(
                audio_file=best_clip,
                speaker_name=speaker_name,
                language=language,
                progress_callback=progress_callback
            )
    
    def _fallback_youtube_extraction(self, youtube_url, speaker_name, language, max_clips, progress_callback):
        """Fallback method using latest yt-dlp with bestaudio format (ChatGPT recommended)"""
        
        try:
            import yt_dlp
        except ImportError:
            if progress_callback:
                progress_callback("❌ yt-dlp required for YouTube voice cloning: pip install yt-dlp")
            return None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cookies_file = Path.cwd() / "cookies.txt"
            
            # Updated fallback extraction with bestaudio format
            output_template = str(temp_path / 'fallback.%(ext)s')
            ydl_opts = {
                'format': 'bestaudio',  # ChatGPT recommendation: use bestaudio
                'outtmpl': output_template,
                'extractaudio': True,
                'audioformat': 'wav',
                'audioquality': 0,  # Best quality
                'quiet': False,
                'no_warnings': False,
                'retries': 10,
                'fragment_retries': 20,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
            }
            
            # Add cookies if available (ChatGPT recommendation)
            if cookies_file.exists():
                ydl_opts['cookiefile'] = str(cookies_file)
                if progress_callback:
                    progress_callback("🍪 Using cookies.txt for enhanced extraction")
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                
                # Look for downloaded file
                for file_path in temp_path.glob('fallback.*'):
                    if file_path.stat().st_size > 1000000:  # > 1MB
                        if progress_callback:
                            size_mb = file_path.stat().st_size / 1024 / 1024
                            progress_callback(f"✅ Fallback extraction successful ({size_mb:.1f}MB)")
                        
                        # Process for voice cloning
                        clips = self._extract_voice_clips(file_path, max_clips, progress_callback)
                        if clips:
                            best_clip = self._select_best_clip(clips)
                            if best_clip:
                                return self.clone_voice_from_audio(
                                    audio_file=best_clip,
                                    speaker_name=speaker_name,
                                    language=language,
                                    progress_callback=progress_callback
                                )
                        break
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ Fallback extraction failed: {str(e)}")
                
                # Try with alternative format as last resort
                if progress_callback:
                    progress_callback("🔄 Trying alternative format...")
                
                ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best[height<=720]/best'
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])
                    
                    # Look for downloaded file again
                    for file_path in temp_path.glob('fallback.*'):
                        if file_path.stat().st_size > 500000:  # > 500KB (lower threshold)
                            if progress_callback:
                                size_mb = file_path.stat().st_size / 1024 / 1024
                                progress_callback(f"✅ Alternative format extraction successful ({size_mb:.1f}MB)")
                            
                            clips = self._extract_voice_clips(file_path, max_clips, progress_callback)
                            if clips:
                                best_clip = self._select_best_clip(clips)
                                if best_clip:
                                    return self.clone_voice_from_audio(
                                        audio_file=best_clip,
                                        speaker_name=speaker_name,
                                        language=language,
                                        progress_callback=progress_callback
                                    )
                            break
                except Exception:
                    pass
        
        return None
    
    def _process_audio_for_cloning(self, audio_file: Path, progress_callback=None) -> Optional[AudioSegment]:
        """Process audio file for optimal voice cloning"""
        try:
            # Load audio
            if audio_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']:
                audio = AudioSegment.from_file(audio_file)
            else:
                if progress_callback:
                    progress_callback(f"⚠️ Unsupported audio format: {audio_file.suffix}")
                return None
            
            # Convert to target sample rate
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)
            
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Check duration
            duration_seconds = len(audio) / 1000.0
            if duration_seconds < self.min_audio_duration:
                if progress_callback:
                    progress_callback(f"⚠️ Audio too short: {duration_seconds:.1f}s (min: {self.min_audio_duration}s)")
                return None
            
            # Trim if too long
            if duration_seconds > self.max_audio_duration:
                audio = audio[:int(self.max_audio_duration * 1000)]
                if progress_callback:
                    progress_callback(f"✂️ Trimmed audio to {self.max_audio_duration}s")
            
            # Normalize volume
            audio = audio.normalize()
            
            # Basic noise reduction (remove very quiet parts)
            # This is a simple approach - more sophisticated noise reduction could be added
            if hasattr(audio, 'strip_silence'):
                audio = audio.strip_silence(silence_len=500, silence_thresh=audio.dBFS-16)
            
            return audio
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Audio processing failed: {str(e)}")
            return None
    
    def _extract_voice_clips(
        self, 
        audio_file: Path, 
        max_clips: int = 5, 
        progress_callback=None
    ) -> List[AudioSegment]:
        """Extract multiple voice clips from a longer audio file"""
        try:
            audio = AudioSegment.from_file(audio_file)
            
            # Convert to mono and normalize
            audio = audio.set_channels(1).normalize()
            
            # Find speech segments (simple approach based on volume)
            # In production, you might want to use more sophisticated VAD
            clips = []
            min_clip_duration = self.min_audio_duration * 1000  # Convert to ms
            max_clip_duration = self.max_audio_duration * 1000
            
            # Split by silence
            silence_thresh = audio.dBFS - 20  # 20 dB below average
            segments = split_on_silence(
                audio,
                min_silence_len=1000,  # 1 second of silence
                silence_thresh=silence_thresh,
                keep_silence=500  # Keep 500ms of silence
            )
            
            for segment in segments:
                if len(segment) >= min_clip_duration:
                    # Trim if too long
                    if len(segment) > max_clip_duration:
                        segment = segment[:max_clip_duration]
                    
                    clips.append(segment)
                    
                    if len(clips) >= max_clips:
                        break
            
            if progress_callback:
                progress_callback(f"📎 Extracted {len(clips)} voice clips")
            
            return clips
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Clip extraction failed: {str(e)}")
            return []
    
    def _select_best_clip(self, clips: List[AudioSegment]) -> Path:
        """Select the best quality clip from extracted clips"""
        if not clips:
            return None
        
        # Simple quality scoring based on duration and volume consistency
        best_clip = None
        best_score = -1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, clip in enumerate(clips):
                # Score based on duration (prefer clips around 10-15 seconds)
                duration_score = 1.0 - abs(len(clip) / 1000.0 - 12.0) / 12.0
                duration_score = max(0, duration_score)
                
                # Score based on volume consistency
                volume_samples = []
                chunk_size = len(clip) // 10  # 10 samples
                for j in range(0, len(clip), chunk_size):
                    chunk = clip[j:j + chunk_size]
                    if len(chunk) > 100:  # Avoid tiny chunks
                        volume_samples.append(chunk.dBFS)
                
                if volume_samples:
                    volume_std = np.std(volume_samples)
                    consistency_score = 1.0 / (1.0 + volume_std / 10.0)  # Lower std = higher score
                else:
                    consistency_score = 0.5
                
                # Combined score
                total_score = (duration_score + consistency_score) / 2
                
                if total_score > best_score:
                    best_score = total_score
                    best_clip_path = temp_path / f"best_clip_{i}.wav"
                    clip.export(best_clip_path, format="wav")
                    best_clip = best_clip_path
        
            # Save the best clip to a permanent location
            if best_clip and best_clip.exists():
                permanent_path = self.output_dir / "temp_best_clip.wav"
                shutil.copy2(best_clip, permanent_path)
                return permanent_path
        
        return None
    
    def _analyze_voice_quality(self, test_audio: Path, reference_audio: Path) -> float:
        """Analyze the quality of the cloned voice (simple implementation)"""
        try:
            # Load both audio files
            test = AudioSegment.from_wav(test_audio)
            ref = AudioSegment.from_wav(reference_audio)
            
            # Simple quality metrics
            
            # 1. Duration similarity (cloned voice should have similar speaking rate)
            expected_duration = len(ref) * 0.8  # Account for different text
            duration_score = 1.0 - abs(len(test) - expected_duration) / expected_duration
            duration_score = max(0, min(1, duration_score))
            
            # 2. Volume similarity
            volume_diff = abs(test.dBFS - ref.dBFS)
            volume_score = 1.0 - min(volume_diff / 20.0, 1.0)  # 20 dB max difference
            
            # 3. Basic spectral similarity (very simplified)
            # In a real implementation, you'd use more sophisticated audio analysis
            spectral_score = 0.7  # Placeholder
            
            # Combine scores
            quality = (duration_score * 0.3 + volume_score * 0.3 + spectral_score * 0.4)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
            return 0.5  # Default score
    
    def synthesize_with_cloned_voice(
        self,
        text: str,
        voice_name: str,
        output_path: Path,
        language: str = "en",
        progress_callback=None
    ) -> bool:
        """Synthesize text using a previously cloned voice"""
        
        if not self.xtts_model:
            if not self.load_xtts_model(progress_callback):
                return False
        
        try:
            # Find the voice model
            voice_dir = self.output_dir / f"xtts_voice_{voice_name.lower().replace(' ', '_')}"
            if not voice_dir.exists():
                if progress_callback:
                    progress_callback(f"❌ Voice model for {voice_name} not found")
                return False
            
            # Load voice config
            config_path = voice_dir / "voice_config.json"
            if not config_path.exists():
                if progress_callback:
                    progress_callback(f"❌ Voice config for {voice_name} not found")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                voice_config = json.load(f)
            
            # Get reference audio path
            reference_audio = voice_dir / voice_config["reference_audio"].split("/")[-1]
            if not reference_audio.exists():
                if progress_callback:
                    progress_callback(f"❌ Reference audio for {voice_name} not found")
                return False
            
            if progress_callback:
                progress_callback(f"🎤 Synthesizing with {voice_name} voice...")
            
            # Synthesize
            self.xtts_model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=str(reference_audio),
                language=language
            )
            
            if progress_callback:
                progress_callback(f"✅ Audio generated with {voice_name} voice")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Synthesis with cloned voice failed: {str(e)}")
            return False
    
    def list_cloned_voices(self) -> List[Dict[str, Any]]:
        """List all available cloned voices"""
        voices = []
        
        for voice_dir in self.output_dir.glob("xtts_voice_*"):
            config_path = voice_dir / "voice_config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        voice_config = json.load(f)
                    voices.append(voice_config)
                except Exception as e:
                    logger.warning(f"Could not load voice config from {config_path}: {e}")
        
        return sorted(voices, key=lambda x: x.get("quality_score", 0), reverse=True)
    
    def delete_voice(self, voice_name: str) -> bool:
        """Delete a cloned voice model"""
        try:
            voice_dir = self.output_dir / f"xtts_voice_{voice_name.lower().replace(' ', '_')}"
            if voice_dir.exists():
                shutil.rmtree(voice_dir)
                return True
            return False
        except Exception:
            return False


# Convenience functions
def clone_voice_from_file(
    audio_file: Path,
    speaker_name: str,
    language: str = "en",
    output_dir: Optional[Path] = None,
    progress_callback=None
) -> Optional[Dict[str, Any]]:
    """Convenience function to clone voice from audio file"""
    
    cloner = XTTSVoiceCloner(output_dir=output_dir)
    return cloner.clone_voice_from_audio(
        audio_file=audio_file,
        speaker_name=speaker_name,
        language=language,
        progress_callback=progress_callback
    )

def clone_voice_from_youtube_url(
    url: str,
    speaker_name: str,
    language: str = "en",
    output_dir: Optional[Path] = None,
    progress_callback=None
) -> Optional[Dict[str, Any]]:
    """Convenience function to clone voice from YouTube"""
    
    cloner = XTTSVoiceCloner(output_dir=output_dir)
    return cloner.clone_voice_from_youtube(
        youtube_url=url,
        speaker_name=speaker_name,
        language=language,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Demo functionality
    print("🎤 Enhanced Voice Cloning with XTTS-v2")
    
    cloner = XTTSVoiceCloner()
    
    print(f"📁 Voice models directory: {cloner.output_dir}")
    print(f"🌍 Supported languages: {', '.join(cloner.supported_languages)}")
    
    # List existing voices
    voices = cloner.list_cloned_voices()
    if voices:
        print("\n🎭 Available cloned voices:")
        for voice in voices:
            print(f"  • {voice['name']} ({voice['language']}) - Quality: {voice['quality_score']:.2f}")
    else:
        print("\n📭 No cloned voices found")
    
    print("\n💡 Use clone_voice_from_file() or clone_voice_from_youtube_url() to create new voices")
