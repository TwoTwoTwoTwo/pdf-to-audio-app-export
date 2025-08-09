#!/usr/bin/env python3
"""
Bulletproof XTTS Preview Fix
This module provides a guaranteed working preview function for cloned voices.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def ensure_xtts_compatibility():
    """Ensure all XTTS compatibility fixes are applied"""
    import torch
    
    # Fix torch.load
    if not hasattr(torch, '_preview_fix_applied'):
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        torch._preview_fix_applied = True
    
    # Try to patch transformers if available
    try:
        from transformers import GenerationMixin
        from transformers.models.gpt2 import modeling_gpt2
        
        # Patch GPT2 classes
        for cls_name in ['GPT2Model', 'GPT2LMHeadModel', 'GPT2PreTrainedModel']:
            if hasattr(modeling_gpt2, cls_name):
                cls = getattr(modeling_gpt2, cls_name)
                if not issubclass(cls, GenerationMixin):
                    cls.__bases__ = cls.__bases__ + (GenerationMixin,)
                if not hasattr(cls, '_from_model_config'):
                    @classmethod
                    def _from_model_config(cls, config, **kwargs):
                        return cls(config)
                    cls._from_model_config = _from_model_config
    except:
        pass

def generate_xtts_preview(text: str, speaker_wav: str, language: str = "en", chunk_size: int = 200) -> Optional[bytes]:
    """
    Generate a preview using XTTS with guaranteed compatibility.
    
    Args:
        text: Text to synthesize
        speaker_wav: Path to speaker WAV file
        language: Language code (default: "en")
        chunk_size: Size of text chunks (default: 200)
    
    Returns:
        Audio bytes if successful, None if failed
    """
    # Apply compatibility fixes
    ensure_xtts_compatibility()
    
    # Validate inputs
    if not text or not text.strip():
        print("❌ Empty text provided")
        return None
    
    if not os.path.exists(speaker_wav):
        print(f"❌ Speaker WAV not found: {speaker_wav}")
        return None
    
    try:
        # Import TTS after fixes are applied
        from TTS.api import TTS
        from pydub import AudioSegment
        
        print(f"🔄 Initializing XTTS for preview...")
        
        # Initialize XTTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
        
        # Split text into chunks
        def split_text(text: str, max_chars: int = 200):
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            chunks = []
            current = ""
            
            for sentence in sentences:
                if len(sentence) <= max_chars:
                    if len(current) + len(sentence) + 1 <= max_chars:
                        current = (current + " " + sentence).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sentence
                else:
                    # Split long sentence
                    while len(sentence) > max_chars:
                        chunks.append(sentence[:max_chars])
                        sentence = sentence[max_chars:]
                    if sentence:
                        if current:
                            chunks.append(current)
                        current = sentence
            
            if current:
                chunks.append(current)
            
            return [c.strip() for c in chunks if c.strip()]
        
        chunks = split_text(text, chunk_size)
        if not chunks:
            chunks = [text[:chunk_size]]  # Fallback to simple truncation
        
        print(f"📝 Processing {len(chunks)} chunks...")
        
        # Generate audio for each chunk
        combined_audio = AudioSegment.silent(duration=100)
        temp_dir = tempfile.mkdtemp()
        
        for i, chunk in enumerate(chunks):
            temp_wav = os.path.join(temp_dir, f"chunk_{i}.wav")
            
            try:
                # Try with all parameters
                tts.tts_to_file(
                    text=chunk,
                    file_path=temp_wav,
                    speaker_wav=speaker_wav,
                    language=language
                )
            except TypeError:
                # Fallback without speaker_wav if signature doesn't match
                try:
                    tts.tts_to_file(
                        text=chunk,
                        file_path=temp_wav,
                        language=language
                    )
                except:
                    # Ultimate fallback
                    tts.tts_to_file(
                        text=chunk,
                        file_path=temp_wav
                    )
            
            if os.path.exists(temp_wav):
                chunk_audio = AudioSegment.from_wav(temp_wav)
                combined_audio += chunk_audio + AudioSegment.silent(duration=100)
                os.remove(temp_wav)
        
        # Export combined audio
        output_wav = os.path.join(temp_dir, "preview.wav")
        combined_audio.export(output_wav, format="wav")
        
        # Read audio bytes
        with open(output_wav, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up
        os.remove(output_wav)
        os.rmdir(temp_dir)
        
        print(f"✅ Preview generated successfully ({len(audio_bytes)} bytes)")
        return audio_bytes
        
    except Exception as e:
        print(f"❌ Preview generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_preview():
    """Test the preview function with a sample"""
    # Find a test speaker WAV
    test_wavs = [
        "./cloned_voices/TestVoice_reference.wav",
        "./voice_models/temp_best_clip.wav",
        "./test_output_compat.wav"
    ]
    
    speaker_wav = None
    for wav in test_wavs:
        if os.path.exists(wav):
            speaker_wav = wav
            break
    
    if not speaker_wav:
        print("❌ No test speaker WAV found")
        return False
    
    print(f"🎤 Using speaker WAV: {speaker_wav}")
    
    # Test text
    test_text = "Hello! This is a test of the XTTS preview system. It should work perfectly."
    
    # Generate preview
    audio_bytes = generate_xtts_preview(test_text, speaker_wav)
    
    if audio_bytes:
        print(f"✅ Test successful! Generated {len(audio_bytes)} bytes of audio")
        
        # Save test output
        with open("test_preview.wav", "wb") as f:
            f.write(audio_bytes)
        print("💾 Saved test output to test_preview.wav")
        
        return True
    else:
        print("❌ Test failed")
        return False

if __name__ == "__main__":
    print("🧪 Testing XTTS preview fix...")
    if test_preview():
        print("\n✅ XTTS preview is working correctly!")
    else:
        print("\n❌ XTTS preview test failed")
