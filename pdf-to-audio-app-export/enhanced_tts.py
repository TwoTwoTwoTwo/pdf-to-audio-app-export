#!/usr/bin/env python3
"""
Enhanced TTS Engine with High-Quality Vocoders and Voice Cloning
Supports LJSpeech+HiFiGAN-v2, XTTS-v2, and VCTK speaker indexing
"""

import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union, Tuple
from TTS.api import TTS
from pydub import AudioSegment
import unicodedata
from datetime import datetime
import logging
import warnings
import os

# Suppress torchaudio backend dispatch warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*par-call.*backend.*dispatch.*")
    warnings.filterwarnings("ignore", message=".*backend.*implementation.*directly.*")
    warnings.filterwarnings("ignore", message=".*I/O functions.*backend.*dispatch.*")
    import torchaudio

# Suppress TTS warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="TTS")
logging.getLogger("TTS").setLevel(logging.ERROR)

# Fix PyTorch 2.6+ compatibility for XTTS-v2 loading
def fix_pytorch_xtts_compatibility():
    """Fix PyTorch compatibility issues with XTTS-v2 model loading"""
    try:
        # Add safe globals for XTTS config and model classes
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
        
        # Add XTTS model classes that might be needed
        try:
            from TTS.tts.models.xtts import XttsAudioConfig
            torch.serialization.add_safe_globals([XttsAudioConfig])
        except ImportError:
            pass
            
        # Also add other common TTS classes that might be needed
        try:
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.configs.tacotron2_config import Tacotron2Config
            torch.serialization.add_safe_globals([VitsConfig, Tacotron2Config])
        except ImportError:
            pass  # These might not exist in all TTS versions
            
    except Exception as e:
        # If we can't fix it, we'll handle it in the loading code
        pass

# PyTorch load patching for XTTS-v2 compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that handles XTTS-v2 compatibility"""
    # For XTTS-v2 models, disable weights_only to avoid loading errors
    if 'weights_only' not in kwargs:
        # Check if this is likely an XTTS model load
        if len(args) > 0 and isinstance(args[0], (str, os.PathLike)):
            path_str = str(args[0])
            if 'xtts' in path_str.lower() or 'multi-dataset' in path_str.lower():
                kwargs['weights_only'] = False
    
    return _original_torch_load(*args, **kwargs)

# Apply the fixes on import
fix_pytorch_xtts_compatibility()
torch.load = _patched_torch_load

# Import and apply the comprehensive XTTS-v2 fix
try:
    from xtts_v2_fix import apply_xtts_v2_transformers_fix, suppress_transformers_warnings
    suppress_transformers_warnings()
    apply_xtts_v2_transformers_fix()
except ImportError:
    print("⚠️ XTTS-v2 fix module not available - some functionality may be limited")

class VoiceConfig:
    """Configuration for different voice types"""
    
    # High-quality vocoder models
    LJSPEECH_HIFIGAN = {
        "tts_model": "tts_models/en/ljspeech/tacotron2-DDC_ph",
        "vocoder_model": "vocoder_models/en/ljspeech/hifigan_v2",
        "sample_rate": 22050,
        "quality": "high",
        "type": "standard"
    }
    
    LJSPEECH_FASTPITCH = {
        "tts_model": "tts_models/en/ljspeech/fast_pitch",
        "vocoder_model": "vocoder_models/en/ljspeech/hifigan_v2", 
        "sample_rate": 22050,
        "quality": "high",
        "type": "standard"
    }
    
    # XTTS-v2 for voice cloning
    XTTS_V2 = {
        "tts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "sample_rate": 24000,
        "quality": "premium",
        "type": "cloning",
        "supports_cloning": True,
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"]
    }
    
    # VCTK multi-speaker with speaker indexing
    VCTK_SPEAKERS = {
        "tts_model": "tts_models/en/vctk/vits",
        "sample_rate": 22050,
        "quality": "high", 
        "type": "multi_speaker",
        "supports_speaker_selection": True,
        # Common VCTK speakers with characteristics
        "speakers": {
            "p225": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English"},
            "p226": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English"},
            "p227": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Deep"},
            "p228": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Soft"},
            "p229": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Clear"},
            "p230": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Warm"},
            "p231": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Professional"},
            "p232": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Narrator"},
            "p233": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Bright"},
            "p234": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Calm"},
            "p236": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Gentle"},
            "p237": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Strong"},
            "p238": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Confident"},
            "p239": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Smooth"},
            "p240": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Authoritative"},
            "p241": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Serious"},
            "p243": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Friendly"},
            "p244": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Relaxed"},
            "p245": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Dynamic"},
            "p246": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Crisp"},
            "p247": {"gender": "male", "age": "adult", "accent": "english", "name": "Male English Rich"},
            "p248": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Elegant"},
            "p249": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Melodic"},
            "p250": {"gender": "female", "age": "adult", "accent": "english", "name": "Female English Expressive"},
        }
    }


class EnhancedTTS:
    """Enhanced TTS engine with high-quality vocoders and voice cloning"""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the enhanced TTS engine"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model = None
        self.current_config = None
        self.model_cache = {}
        
        # Voice assignment tracking
        self.speaker_voice_map = {}
        self.voice_usage_count = {}
        
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available high-quality TTS models"""
        return {
            "xtts_v2": VoiceConfig.XTTS_V2,  # Prioritize XTTS-v2
            "vctk_multi": VoiceConfig.VCTK_SPEAKERS,
            "ljspeech_hifigan": VoiceConfig.LJSPEECH_HIFIGAN,
            "ljspeech_fastpitch": VoiceConfig.LJSPEECH_FASTPITCH
        }
    
    def get_vctk_speakers(self) -> Dict[str, Dict[str, str]]:
        """Get available VCTK speakers with their characteristics"""
        return VoiceConfig.VCTK_SPEAKERS["speakers"]
    
    def load_model(self, model_key: str, progress_callback: Optional[Callable] = None) -> bool:
        """Load a specific TTS model"""
        try:
            models = self.get_available_models()
            if model_key not in models:
                raise ValueError(f"Model {model_key} not found")
            
            config = models[model_key]
            
            if progress_callback:
                progress_callback(f"🔄 Loading {model_key} model...")
            
            # Check cache first
            if model_key in self.model_cache:
                self.current_model = self.model_cache[model_key]
                self.current_config = config
                if progress_callback:
                    progress_callback(f"✅ Model {model_key} loaded from cache")
                return True
            
            # Load new model with fallback handling
            tts = None
            
            if config["type"] == "cloning":
                # XTTS-v2 for voice cloning - handle PyTorch loading issues
                if progress_callback:
                    progress_callback(f"🔄 Loading XTTS-v2 for voice cloning...")
                try:
                    # Try with safe loading first
                    tts = TTS(config["tts_model"])
                    if progress_callback:
                        progress_callback(f"✅ XTTS-v2 loaded successfully")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ XTTS-v2 failed ({str(e)[:100]}...), falling back to VCTK")
                    # Fallback to VCTK for voice cloning attempts
                    try:
                        tts = TTS("tts_models/en/vctk/vits")
                        # Update config to reflect the fallback
                        config = VoiceConfig.VCTK_SPEAKERS.copy()
                        if progress_callback:
                            progress_callback(f"✅ Fallback to VCTK multi-speaker successful")
                    except Exception as e2:
                        if progress_callback:
                            progress_callback(f"❌ All voice cloning models failed: {e2}")
                        raise e2
                        
            elif config["type"] == "multi_speaker":
                # VCTK multi-speaker
                if progress_callback:
                    progress_callback(f"🔄 Loading VCTK multi-speaker model...")
                tts = TTS(config["tts_model"])
                
            else:
                # Standard models - use simple model loading (TTS handles vocoder automatically)
                if progress_callback:
                    progress_callback(f"🔄 Loading {config['tts_model']}...")
                tts = TTS(config["tts_model"])
            
            if tts and self.device != "cpu":
                try:
                    tts = tts.to(self.device)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ Could not move to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            # Cache the model
            self.model_cache[model_key] = tts
            self.current_model = tts
            self.current_config = config
            
            if progress_callback:
                progress_callback(f"✅ Model {model_key} loaded successfully")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Failed to load model {model_key}: {str(e)}")
            return False
    
    def assign_speakers_to_voices(
        self, 
        speakers: Dict[str, Any], 
        preferred_model: str = "vctk_multi"
    ) -> Dict[str, str]:
        """Intelligently assign speakers to available voices"""
        
        if not speakers:
            return {}
        
        assignments = {}
        available_voices = list(VoiceConfig.VCTK_SPEAKERS["speakers"].keys())
        voice_characteristics = VoiceConfig.VCTK_SPEAKERS["speakers"]
        
        # Sort speakers by importance (dialogue count, word count)
        sorted_speakers = sorted(
            speakers.items(),
            key=lambda x: (
                getattr(x[1], 'dialogue_count', 0) + 
                getattr(x[1], 'word_count', 0) / 100
            ),
            reverse=True
        )
        
        assigned_voices = set()
        
        for speaker_name, speaker_data in sorted_speakers:
            best_voice = None
            
            # Get speaker characteristics
            characteristics = getattr(speaker_data, 'characteristics', {})
            formality = characteristics.get('formality_score', 0.5)
            excitement = characteristics.get('excitement_level', 0.3)
            calmness = characteristics.get('calmness_level', 0.5)
            
            # Try to match gender if we can infer it
            likely_gender = self._infer_gender_from_name(speaker_name)
            
            # Find best matching voice
            for voice_id in available_voices:
                if voice_id in assigned_voices:
                    continue
                    
                voice_info = voice_characteristics[voice_id]
                score = 0
                
                # Gender matching (highest priority)
                if likely_gender and voice_info["gender"] == likely_gender:
                    score += 10
                
                # Characteristic matching
                if formality > 0.7:
                    if "professional" in voice_info["name"].lower() or \
                       "authoritative" in voice_info["name"].lower():
                        score += 5
                elif excitement > 0.6:
                    if "bright" in voice_info["name"].lower() or \
                       "dynamic" in voice_info["name"].lower():
                        score += 5
                elif calmness > 0.6:
                    if "calm" in voice_info["name"].lower() or \
                       "gentle" in voice_info["name"].lower() or \
                       "soft" in voice_info["name"].lower():
                        score += 5
                
                # Narrator gets specific voices
                if speaker_name.lower() == "narrator":
                    if "narrator" in voice_info["name"].lower() or \
                       "clear" in voice_info["name"].lower():
                        score += 8
                
                if not best_voice or score > best_voice[1]:
                    best_voice = (voice_id, score)
            
            # Assign the best voice or fallback
            if best_voice:
                assignments[speaker_name] = best_voice[0]
                assigned_voices.add(best_voice[0])
            else:
                # Fallback to any available voice
                remaining = [v for v in available_voices if v not in assigned_voices]
                if remaining:
                    assignments[speaker_name] = remaining[0]
                    assigned_voices.add(remaining[0])
        
        return assignments
    
    def _infer_gender_from_name(self, name: str) -> Optional[str]:
        """Simple gender inference from speaker names"""
        name_lower = name.lower()
        
        # Common patterns
        female_indicators = [
            'mrs', 'ms', 'miss', 'lady', 'woman', 'girl', 'mother', 'mom',
            'sister', 'daughter', 'aunt', 'grandmother', 'queen', 'princess'
        ]
        
        male_indicators = [
            'mr', 'sir', 'lord', 'man', 'boy', 'father', 'dad',
            'brother', 'son', 'uncle', 'grandfather', 'king', 'prince'
        ]
        
        # Common first names (simplified)
        female_names = [
            'mary', 'sarah', 'emma', 'olivia', 'ava', 'isabella', 'sophia',
            'charlotte', 'mia', 'amelia', 'harper', 'evelyn', 'abigail',
            'emily', 'elizabeth', 'sofia', 'madison', 'avery', 'ella', 'scarlett'
        ]
        
        male_names = [
            'james', 'robert', 'john', 'michael', 'david', 'william', 'richard',
            'joseph', 'thomas', 'christopher', 'charles', 'daniel', 'matthew',
            'anthony', 'mark', 'donald', 'steven', 'paul', 'joshua', 'andrew'
        ]
        
        # Check for indicators
        for indicator in female_indicators:
            if indicator in name_lower:
                return "female"
                
        for indicator in male_indicators:
            if indicator in name_lower:
                return "male"
        
        # Check for names
        for fname in female_names:
            if fname in name_lower:
                return "female"
                
        for mname in male_names:
            if mname in name_lower:
                return "male"
        
        return None
    
    def synthesize_text(
        self,
        text: str,
        output_path: Path,
        speaker_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        progress_callback: Optional[Callable] = None,
        voice_config: Optional[Dict] = None
    ) -> bool:
        """Synthesize text with the current model, supporting cloned voices"""
        
        if not self.current_model:
            if progress_callback:
                progress_callback("❌ No model loaded")
            return False
        
        try:
            if progress_callback:
                progress_callback(f"🔊 Synthesizing {len(text)} characters...")
            
            # Handle cloned voices with special logic
            if voice_config and voice_config.get('type') == 'cloned' and voice_config.get('speaker_wav'):
                # Use XTTS-v2 for cloned voices regardless of current model
                try:
                    cloning_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
                    cloning_model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        speaker_wav=voice_config['speaker_wav'],
                        language=language
                    )
                    if progress_callback:
                        progress_callback(f"✅ Cloned voice audio saved to {output_path.name}")
                    return True
                    
                except Exception as xtts_error:
                    if progress_callback:
                        progress_callback(f"⚠️ XTTS-v2 failed, trying YourTTS fallback: {xtts_error}")
                    
                    # Fallback to YourTTS
                    try:
                        your_tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
                        your_tts_model.tts_to_file(
                            text=text,
                            file_path=str(output_path),
                            speaker_wav=voice_config['speaker_wav'],
                            language=language
                        )
                        if progress_callback:
                            progress_callback(f"✅ YourTTS cloned voice audio saved to {output_path.name}")
                        return True
                        
                    except Exception as your_error:
                        if progress_callback:
                            progress_callback(f"❌ Both cloning models failed. Using regular voice: {your_error}")
                        # Continue with regular voice synthesis below
            
            # Handle different model types for regular voices
            if self.current_config["type"] == "cloning" and speaker_wav:
                # XTTS-v2 voice cloning
                self.current_model.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=speaker_wav,
                    language=language
                )
                
            elif self.current_config["type"] == "multi_speaker" and speaker_id:
                # VCTK multi-speaker
                self.current_model.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker=speaker_id
                )
                
            else:
                # Standard TTS
                self.current_model.tts_to_file(
                    text=text,
                    file_path=str(output_path)
                )
            
            if progress_callback:
                progress_callback(f"✅ Audio saved to {output_path.name}")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Synthesis failed: {str(e)}")
            return False
    
    def synthesize_chunks_enhanced(
        self,
        chunks: List[str],
        output_path: Path,
        speaker_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """Enhanced chunk synthesis with better audio processing"""
        
        if not chunks:
            return False
        
        try:
            combined = AudioSegment.silent(duration=300)  # Shorter initial silence
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # Skip very short chunks
                    continue
                
                temp_path = output_path.parent / f"{output_path.stem}_chunk_{i:03d}.wav"
                
                if progress_callback:
                    progress_callback(f"🔊 Processing chunk {i+1}/{len(chunks)}")
                
                if self.synthesize_text(
                    text=chunk.strip(),
                    output_path=temp_path,
                    speaker_id=speaker_id,
                    speaker_wav=speaker_wav,
                    language=language
                ):
                    # Load and process the audio
                    try:
                        audio = AudioSegment.from_wav(temp_path)
                        
                        # Apply audio improvements
                        audio = self._enhance_audio_quality(audio)
                        
                        # Add to combined audio with appropriate spacing
                        if i > 0:
                            combined += AudioSegment.silent(duration=400)  # Pause between chunks
                        combined += audio
                        
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"⚠️ Could not process chunk {i}: {e}")
                    finally:
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
            
            # Export final audio
            # Check if we have meaningful audio content (more than just the initial silence)
            if len(combined) > 800:  # More than just initial silence (300ms) + some buffer
                if progress_callback:
                    progress_callback(f"✅ Generated {len(combined)/1000:.1f}s of audio")
                combined.export(output_path, format="mp3", bitrate="128k")
                return True
            else:
                if progress_callback:
                    progress_callback(f"❌ Generated audio too short: {len(combined)/1000:.1f}s")
                return False
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Chunk synthesis failed: {str(e)}")
            return False
    
    def _enhance_audio_quality(self, audio: AudioSegment) -> AudioSegment:
        """Apply audio enhancements for better quality"""
        try:
            # Normalize volume
            audio = audio.normalize()
            
            # Apply gentle compression to reduce dynamic range
            # This helps with consistency across different chunks
            if hasattr(audio, 'compress_dynamic_range'):
                audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            
            # Remove clicks and pops (basic implementation)
            # In a real implementation, you might use more sophisticated audio processing
            
            return audio
            
        except Exception as e:
            # If enhancement fails, return original audio
            return audio
    
    def convert_multi_speaker_audiobook(
        self,
        chapters_dir: Path,
        output_dir: Path,
        speaker_assignments: Dict[str, str],
        model_key: str = "vctk_multi",
        progress_callback: Optional[Callable] = None
    ) -> List[Path]:
        """Convert entire audiobook with multi-speaker support"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.load_model(model_key, progress_callback):
            return []
        
        # Get text files
        txt_files = sorted(list(chapters_dir.glob("*.txt")))
        if not txt_files:
            if progress_callback:
                progress_callback("❌ No text files found")
            return []
        
        created_files = []
        
        for txt_file in txt_files:
            try:
                if progress_callback:
                    progress_callback(f"📖 Processing {txt_file.name}")
                
                text = txt_file.read_text(encoding='utf-8').strip()
                text = self._normalize_text(text)
                
                # Split into chunks
                chunks = self._split_text_smart(text)
                
                if not chunks:
                    continue
                
                # Determine speaker for this chapter
                # For now, use default speaker unless specified
                speaker_id = speaker_assignments.get("Narrator", list(speaker_assignments.values())[0])
                
                # Generate audio
                output_file = output_dir / f"{txt_file.stem}.mp3"
                
                if self.synthesize_chunks_enhanced(
                    chunks=chunks,
                    output_path=output_file,
                    speaker_id=speaker_id,
                    progress_callback=progress_callback
                ):
                    created_files.append(output_file)
                    if progress_callback:
                        progress_callback(f"✅ Created {output_file.name}")
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ Error processing {txt_file.name}: {e}")
                continue
        
        return created_files
    
    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization for better TTS"""
        # Basic normalization
        replacements = {
            """: '"', """: '"', "'": "'", "'": "'",
            "—": "-", "–": "-", "…": "...",
            "\u00a0": " ",  # non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        text = unicodedata.normalize("NFKC", text)
        
        # Handle specific time abbreviations that should be spoken as letters (do this first)
        text = re.sub(r'\bA\.M\.?\b', 'A M', text, flags=re.IGNORECASE)
        text = re.sub(r'\bP\.M\.?\b', 'P M', text, flags=re.IGNORECASE) 
        text = re.sub(r'\bAM\b', 'A M', text)
        text = re.sub(r'\bPM\b', 'P M', text)
        
        # Handle common country/organization abbreviations that should be letters
        text = re.sub(r'\bU\.S\.A\.?\b', 'U S A', text, flags=re.IGNORECASE)
        text = re.sub(r'\bU\.K\.?\b', 'U K', text, flags=re.IGNORECASE)
        text = re.sub(r'\bF\.B\.I\.?\b', 'F B I', text, flags=re.IGNORECASE)
        text = re.sub(r'\bC\.I\.A\.?\b', 'C I A', text, flags=re.IGNORECASE)
        text = re.sub(r'\bN\.A\.S\.A\.?\b', 'N A S A', text, flags=re.IGNORECASE)
        
        # Handle variations without periods
        text = re.sub(r'\bUSA\b', 'U S A', text)
        text = re.sub(r'\bUK\b', 'U K', text) 
        text = re.sub(r'\bFBI\b', 'F B I', text)
        text = re.sub(r'\bCIA\b', 'C I A', text)
        text = re.sub(r'\bNASA\b', 'N A S A', text)
        
        # Handle degree abbreviations
        text = re.sub(r'\bPh\.?D\.?\b', 'P H D', text, flags=re.IGNORECASE)
        text = re.sub(r'\bM\.?D\.?\b', 'M D', text, flags=re.IGNORECASE)
        text = re.sub(r'\bM\.?A\.?\b', 'M A', text, flags=re.IGNORECASE)
        text = re.sub(r'\bB\.?A\.?\b', 'B A', text, flags=re.IGNORECASE)
        
        # Fix common title abbreviations (expand these instead of letters)
        text = re.sub(r'\bDr\.', 'Doctor', text)
        text = re.sub(r'\bMr\.', 'Mister', text)
        text = re.sub(r'\bMrs\.', 'Missus', text) 
        text = re.sub(r'\bMs\.', 'Miss', text)
        text = re.sub(r'\bProf\.', 'Professor', text)
        text = re.sub(r'\bSt\.', 'Saint', text)
        text = re.sub(r'\bAve\.', 'Avenue', text)
        text = re.sub(r'\bRd\.', 'Road', text)
        
        # Handle measurements and units (expand for clarity)
        text = re.sub(r'\b(\d+)\s*ft\.?\b', r'\1 feet', text)
        text = re.sub(r'\b(\d+)\s*in\.?\b', r'\1 inches', text)
        text = re.sub(r'\b(\d+)\s*lbs?\.?\b', r'\1 pounds', text)
        text = re.sub(r'\b(\d+)\s*kg\.?\b', r'\1 kilograms', text)
        text = re.sub(r'\b(\d+)\s*mph\b', r'\1 miles per hour', text)
        text = re.sub(r'\b(\d+)\s*km/h\b', r'\1 kilometers per hour', text)
        
        # Handle ordinal numbers
        text = re.sub(r'\b(\d+)st\b', r'\1st', text)  # Keep ordinals as is for now
        text = re.sub(r'\b(\d+)nd\b', r'\1nd', text)
        text = re.sub(r'\b(\d+)rd\b', r'\1rd', text)
        text = re.sub(r'\b(\d+)th\b', r'\1th', text)
        
        # Fix numbers and years (improved)
        # Handle years more naturally
        text = re.sub(r'\b19(\d{2})\b', lambda m: f"nineteen {m.group(1)}", text)
        text = re.sub(r'\b20(\d{2})\b', lambda m: f"twenty {m.group(1).zfill(2)}", text)
        
        # Handle large numbers
        text = re.sub(r'\b(\d{1,3}),(\d{3})\b', r'\1 thousand \2', text)
        
        # Handle single capital letters that might be initials (but not at start of sentence)
        # This is tricky - we want "M. Frank" to be "M Frank" but "I went" to stay "I went"
        # Use a simpler pattern that handles common initial patterns
        text = re.sub(r'\b([A-Z])\. ([A-Z][a-z]+)\b', r'\1 \2', text)
        
        # Clean up multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_text_smart(self, text: str, max_length: int = 5000, min_length: int = 30) -> List[str]:
        """Smart text splitting optimized for TTS"""
        
        # First split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is short enough, add to current chunk
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it's substantial
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk)
                
                # If paragraph is too long, split by sentences
                if len(paragraph) > max_length:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 <= max_length:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                        else:
                            if len(temp_chunk) >= min_length:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        
        # Don't forget the last chunk
        if len(current_chunk) >= min_length:
            chunks.append(current_chunk)
        
        return chunks


def create_enhanced_tts_config(output_path: Path) -> Dict[str, Any]:
    """Create a configuration file for enhanced TTS settings"""
    
    config = {
        "version": "2.0",
        "created": datetime.now().isoformat(),
        "models": {
            "high_quality": {
                "ljspeech_hifigan": VoiceConfig.LJSPEECH_HIFIGAN,
                "ljspeech_fastpitch": VoiceConfig.LJSPEECH_FASTPITCH
            },
            "voice_cloning": {
                "xtts_v2": VoiceConfig.XTTS_V2
            },
            "multi_speaker": {
                "vctk": VoiceConfig.VCTK_SPEAKERS
            }
        },
        "recommended_settings": {
            "single_speaker_high_quality": "ljspeech_hifigan",
            "voice_cloning": "xtts_v2", 
            "multi_speaker": "vctk",
            "fastest": "ljspeech_fastpitch"
        },
        "audio_settings": {
            "sample_rate": 22050,
            "bitrate": "128k",
            "format": "mp3",
            "silence_duration_ms": 400,
            "normalize_audio": True
        }
    }
    
    # Save config
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config


# Convenience functions for backward compatibility
def convert_txt_to_mp3_enhanced(
    input_dir: Path,
    output_dir: Path,
    model_key: str = "ljspeech_hifigan",
    speaker_assignments: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable] = None
) -> List[Path]:
    """Enhanced conversion function with high-quality vocoders"""
    
    tts_engine = EnhancedTTS()
    
    if speaker_assignments:
        return tts_engine.convert_multi_speaker_audiobook(
            chapters_dir=input_dir,
            output_dir=output_dir,
            speaker_assignments=speaker_assignments,
            model_key=model_key,
            progress_callback=progress_callback
        )
    else:
        # Single speaker mode
        if not tts_engine.load_model(model_key, progress_callback):
            return []
        
        txt_files = sorted(list(input_dir.glob("*.txt")))
        created_files = []
        
        for txt_file in txt_files:
            text = txt_file.read_text(encoding='utf-8').strip()
            text = tts_engine._normalize_text(text)
            chunks = tts_engine._split_text_smart(text)
            
            output_file = output_dir / f"{txt_file.stem}.mp3"
            
            # Use default speaker for VCTK multi-speaker model
            speaker_id = None
            if model_key == 'vctk_multi':
                speaker_id = 'p225'  # Default reliable female speaker
            
            if tts_engine.synthesize_chunks_enhanced(
                chunks=chunks,
                output_path=output_file,
                speaker_id=speaker_id,
                progress_callback=progress_callback
            ):
                created_files.append(output_file)
        
        return created_files


if __name__ == "__main__":
    # Demo/test functionality
    print("🎙️ Enhanced TTS Engine Demo")
    
    tts = EnhancedTTS()
    models = tts.get_available_models()
    
    print("\n📋 Available Models:")
    for key, config in models.items():
        print(f"  • {key}: {config['quality']} quality, {config['type']} type")
    
    print("\n🎤 VCTK Speakers (sample):")
    speakers = tts.get_vctk_speakers()
    for speaker_id, info in list(speakers.items())[:5]:
        print(f"  • {speaker_id}: {info['name']} ({info['gender']})")
    
    # Create example config
    config_path = Path("enhanced_tts_config.json")
    config = create_enhanced_tts_config(config_path)
    print(f"\n📄 Config saved to {config_path}")
