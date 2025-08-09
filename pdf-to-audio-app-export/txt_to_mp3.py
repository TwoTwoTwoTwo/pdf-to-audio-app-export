# Use the direct XTTS implementation (best compatibility)
try:
    from direct_xtts import get_direct_xtts, direct_tts_to_file
    DIRECT_XTTS_AVAILABLE = True
    print("✅ Direct XTTS implementation available (highest compatibility)")
except ImportError:
    DIRECT_XTTS_AVAILABLE = False
    get_direct_xtts = None
    direct_tts_to_file = None
    print("❌ Direct XTTS not available")

# Fallback: Use the final working XTTS implementation
try:
    from xtts_final_working import get_final_working_tts, generate_tts_audio
    XTTS_WORKING = True
    print("✅ Final working XTTS implementation available for TTS conversion")
except ImportError:
    XTTS_WORKING = False
    get_final_working_tts = None
    generate_tts_audio = None
    print("❌ Final working XTTS implementation not found - using fallback TTS")

import re
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from TTS.api import TTS
from pydub import AudioSegment
import unicodedata
import numpy as np
import torch
import json
import os
from datetime import datetime


def normalize_text(text: str) -> str:
    """
    Normalize smart quotes and unusual unicode characters.
    """
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "–": "-",
        "\u00a0": " ",  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return unicodedata.normalize("NFKC", text)


def split_into_chunks(text: str, max_length: int = 4800, min_length: int = 20) -> List[str]:
    """
    Splits long text into smaller chunks and skips tiny ones that cause kernel errors.
    """
    sentences = re.split(r'(?<=[.?!]) +', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_length:
            current += sentence + " "
        else:
            if len(current.strip()) >= min_length:
                chunks.append(current.strip())
            current = sentence + " "

    if len(current.strip()) >= min_length:
        chunks.append(current.strip())

    return chunks


def synthesize_chunks_to_mp3(chunks: List[str], output_path: Path, tts: TTS, speaker_wav: Optional[str] = None, speaker_id: Optional[str] = None):
    """
    Generate MP3 by combining individual chunk audio outputs.
    Supports cloned voices via speaker_wav parameter and multi-speaker models via speaker_id.
    """
    combined = AudioSegment.silent(duration=500)

    for i, chunk in enumerate(chunks):
        try:
            temp_path = output_path.parent / f"{output_path.stem}_part{i}.wav"
            
            # Use speaker_wav for cloned voices if available
            if speaker_wav and Path(speaker_wav).exists():
                tts.tts_to_file(text=chunk, file_path=str(temp_path), speaker_wav=speaker_wav)
            elif speaker_id:  # Multi-speaker model
                tts.tts_to_file(text=chunk, file_path=str(temp_path), speaker=speaker_id)
            else:
                tts.tts_to_file(text=chunk, file_path=str(temp_path))
            
            audio = AudioSegment.from_wav(temp_path)
            combined += audio + AudioSegment.silent(duration=500)
            temp_path.unlink()  # Clean up temp file
        except Exception as e:
            print(f"⚠️ Skipped chunk {i} due to error: {e}")

    combined.export(output_path, format="mp3")


def _split_for_xtts(text: str, max_chars: int = 220) -> List[str]:
    """
    Split text into chunks at sentence boundaries for XTTS.
    Uses improved chunking to prevent mid-sentence cuts.
    """
    try:
        from improved_text_chunking import split_text_on_sentences, prepare_text_for_tts
        
        # Clean the text first
        text = prepare_text_for_tts(text)
        
        # Use improved sentence-aware chunking
        # XTTS works best with 200-250 char chunks
        chunks = split_text_on_sentences(
            text, 
            max_chunk_size=max_chars,
            min_chunk_size=30,  # Avoid very small chunks
            overlap_sentences=0  # No overlap for now
        )
        
        return chunks if chunks else [text]
        
    except ImportError:
        # Fallback to original implementation if improved chunking not available
        import re
        text = text.strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
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
                # Hard wrap long sentence at word boundaries
                words = s.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= max_chars:
                        temp = (temp + " " + word).strip()
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = word
                if temp:
                    if current:
                        chunks.append(current)
                    current = temp
        if current:
            chunks.append(current)
        return [c.strip() for c in chunks if c.strip()]


def convert_with_final_xtts(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    cloned_voices_data: Optional[dict] = None,
    xtts_settings: Optional[dict] = None,
) -> List[Path]:
    """
    Convert text files to MP3 using XTTS-v2 with stable per-chunk synthesis.
    Uses the standard XTTS API (tts_to_file) with small chunks and speaker_wav for cloned voices.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Initialize XTTS API model (same path previews use)
    if progress_callback:
        progress_callback("🔄 Initializing XTTS-v2 (per-chunk API)...")
    try:
        xtts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Failed to initialize XTTS API: {e}")
        return []
    if progress_callback:
        progress_callback("✅ XTTS API ready for synthesis")
    
    # Get all text files
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        if progress_callback:
            progress_callback("❌ No text files found")
        return []
    
    # Process each text file
    for i, txt_file in enumerate(txt_files):
        chapter_name = txt_file.stem
        
        if progress_callback:
            progress_callback(f"🎵 Converting {chapter_name} ({i+1}/{len(txt_files)})...")
        
        try:
            # Read and normalize text
            text = txt_file.read_text(encoding="utf-8").strip()
            text = normalize_text(text)
            
            if not text:
                if progress_callback:
                    progress_callback(f"⚠️ Skipping empty file: {chapter_name}")
                continue
            
            # Create output file path
            mp3_file = output_dir / f"{chapter_name}.mp3"
            
            # Determine speaker_wav for cloned voices using the new voice cloning fix
            speaker_wav = None
            
            try:
                from voice_cloning_fix import get_voice_for_synthesis, debug_voice_mapping, is_cloned_voice
                
                # Debug the voice mapping
                if progress_callback:
                    progress_callback(f"🔍 Analyzing voice: {model_name}")
                
                debug_voice_mapping(model_name, cloned_voices_data)
                
                # Get the speaker_wav if this is a cloned voice
                speaker_wav = get_voice_for_synthesis(model_name, cloned_voices_data)
                
                if speaker_wav:
                    if progress_callback:
                        progress_callback(f"🎤 Using cloned voice: {speaker_wav}")
                elif is_cloned_voice(model_name):
                    if progress_callback:
                        progress_callback(f"⚠️ Cloned voice '{model_name}' not found, using default synthesis")
                        
            except ImportError:
                # Fallback to old logic if voice_cloning_fix not available
                if "(Cloned)" in model_name and cloned_voices_data:
                    voice_name = model_name.replace("🎬 ", "").replace(" (Cloned)", "")
                    for cloned_name, cloned_data in cloned_voices_data.items():
                        if voice_name.lower() in cloned_name.lower() or cloned_name.lower() in voice_name.lower():
                            speaker_wav = cloned_data.get('speaker_wav')
                            if speaker_wav and Path(speaker_wav).exists():
                                if progress_callback:
                                    progress_callback(f"🎤 Using cloned voice: {voice_name}")
                                break
            
            # Split text into small XTTS-friendly chunks (configurable)
            max_chars = 250  # Increased for better sentence completion
            if xtts_settings and isinstance(xtts_settings.get('chunk_size'), int):
                max_chars = max(100, min(400, xtts_settings['chunk_size']))
            chunks = _split_for_xtts(text, max_chars=max_chars)
            
            # Validate chunks to ensure no text is lost
            try:
                from improved_text_chunking import validate_chunks
                is_valid, validation_msg = validate_chunks(chunks, text)
                if not is_valid:
                    if progress_callback:
                        progress_callback(f"⚠️ Chunk validation issue: {validation_msg}")
            except:
                pass  # Continue anyway if validation not available
            
            if not chunks:
                if progress_callback:
                    progress_callback(f"⚠️ No valid text chunks for: {chapter_name}")
                continue
            
            # Initialize voice consistency manager for cloned voices
            voice_manager = None
            if speaker_wav:
                try:
                    from improved_text_chunking import VoiceConsistencyManager
                    voice_manager = VoiceConsistencyManager(speaker_wav)
                    if progress_callback:
                        progress_callback(f"🎯 Voice consistency manager initialized")
                except:
                    pass
            
            # Generate audio for each chunk and combine
            combined_audio = AudioSegment.silent(duration=300)  # Shorter initial silence
            
            # Track successful chunks for validation
            successful_chunks = 0
            
            for j, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(f"   Processing chunk {j+1}/{len(chunks)} ({len(chunk)} chars)...")
                
                # Create temporary file for this chunk
                temp_wav = output_dir / f"temp_{chapter_name}_chunk_{j}.wav"
                
                try:
                    # Generate audio using XTTS API per chunk (stable)
                    # Build kwargs with optional sampling params, fallback if unsupported
                    _kwargs = {
                        "language": (xtts_settings.get('language') if xtts_settings else "en") or "en",
                    }
                    
                    # Use consistent voice conditioning
                    if speaker_wav and Path(speaker_wav).exists():
                        _kwargs["speaker_wav"] = speaker_wav
                        
                    # Add XTTS settings for voice consistency
                    if isinstance(xtts_settings, dict):
                        # Use more conservative settings for consistency
                        _kwargs["temperature"] = xtts_settings.get("temperature", 0.65)  # Lower for consistency
                        _kwargs["top_k"] = xtts_settings.get("top_k", 50)
                        _kwargs["top_p"] = xtts_settings.get("top_p", 0.85)
                        _kwargs["length_penalty"] = xtts_settings.get("length_penalty", 1.0)
                        _kwargs["repetition_penalty"] = xtts_settings.get("repetition_penalty", 2.0)
                        _kwargs["speed"] = xtts_settings.get("speed", 1.0)
                    
                    try:
                        xtts_api.tts_to_file(text=chunk, file_path=str(temp_wav), **_kwargs)
                    except TypeError:
                        # Fallback to minimal parameters if some aren't supported
                        safe_keys = {"language", "speaker_wav"}
                        xtts_api.tts_to_file(text=chunk, file_path=str(temp_wav), **{k: v for k, v in _kwargs.items() if k in safe_keys})
                    
                    if temp_wav.exists():
                        # Load and add to combined audio
                        chunk_audio = AudioSegment.from_wav(str(temp_wav))
                        
                        # Add shorter silence between chunks for natural flow
                        # Less silence = more natural speech flow
                        silence_duration = 200 if j < len(chunks) - 1 else 300
                        combined_audio += chunk_audio + AudioSegment.silent(duration=silence_duration)
                        
                        successful_chunks += 1
                        
                        # Clean up temp file
                        temp_wav.unlink()
                    else:
                        if progress_callback:
                            progress_callback(f"⚠️ Failed to generate chunk {j+1}")
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ Error with chunk {j+1}: {e}")
                    # Clean up temp file if it exists
                    if temp_wav.exists():
                        temp_wav.unlink()
            
            # Validate that we processed all chunks
            if successful_chunks < len(chunks):
                if progress_callback:
                    progress_callback(f"⚠️ Only {successful_chunks}/{len(chunks)} chunks succeeded")
            
            # Export combined audio as MP3
            if len(combined_audio) > 1000:  # Only save if we have substantial audio
                # Apply normalization for consistent volume
                combined_audio = combined_audio.normalize()
                
                combined_audio.export(str(mp3_file), format="mp3", bitrate="192k")
                created_files.append(mp3_file)
                
                if progress_callback:
                    progress_callback(f"✅ Saved: {mp3_file.name} ({len(combined_audio)/1000:.1f}s, {successful_chunks} chunks)")
            else:
                if progress_callback:
                    progress_callback(f"⚠️ No audio generated for: {chapter_name}")
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {chapter_name}: {e}")
            continue
    
    if progress_callback:
        progress_callback(f"🎉 Final XTTS conversion complete! Created {len(created_files)} files.")
    
    return created_files


def convert_with_direct_xtts(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    cloned_voices_data: Optional[dict] = None,
) -> List[Path]:
    """
    Convert text files to MP3 using the direct XTTS implementation (highest compatibility)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Get the direct XTTS instance
    if progress_callback:
        progress_callback("🔄 Initializing Direct XTTS (highest compatibility)...")
    
    xtts = get_direct_xtts()
    if not xtts:
        if progress_callback:
            progress_callback("❌ Failed to get Direct XTTS instance")
        return []
    
    # Get all text files
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        if progress_callback:
            progress_callback("❌ No text files found")
        return []
    
    # Process each text file
    for i, txt_file in enumerate(txt_files):
        chapter_name = txt_file.stem
        
        if progress_callback:
            progress_callback(f"🎵 Converting {chapter_name} ({i+1}/{len(txt_files)}) with Direct XTTS...")
        
        try:
            # Read and normalize text
            text = txt_file.read_text(encoding="utf-8").strip()
            text = normalize_text(text)
            
            if not text:
                if progress_callback:
                    progress_callback(f"⚠️ Skipping empty file: {chapter_name}")
                continue
            
            # Create output file path
            mp3_file = output_dir / f"{chapter_name}.mp3"
            
            # Determine speaker_wav for cloned voices
            speaker_wav = None
            
            try:
                from voice_cloning_fix import get_voice_for_synthesis, is_cloned_voice
                
                # Get the speaker_wav if this is a cloned voice
                speaker_wav = get_voice_for_synthesis(model_name, cloned_voices_data)
                
                if speaker_wav:
                    if progress_callback:
                        progress_callback(f"🎤 Using cloned voice: {speaker_wav}")
                elif is_cloned_voice(model_name):
                    if progress_callback:
                        progress_callback(f"⚠️ Cloned voice '{model_name}' not found, using default synthesis")
                        
            except ImportError:
                # Fallback logic
                if "(Cloned)" in model_name and cloned_voices_data:
                    voice_name = model_name.replace("🎬 ", "").replace(" (Cloned)", "")
                    for cloned_name, cloned_data in cloned_voices_data.items():
                        if voice_name.lower() in cloned_name.lower():
                            speaker_wav = cloned_data.get('speaker_wav')
                            if speaker_wav and Path(speaker_wav).exists():
                                if progress_callback:
                                    progress_callback(f"🎤 Using cloned voice: {voice_name}")
                                break
            
            # Split text into manageable chunks
            chunks = split_into_chunks(text, max_length=1500)  # Even smaller chunks for Direct XTTS
            
            if not chunks:
                if progress_callback:
                    progress_callback(f"⚠️ No valid text chunks for: {chapter_name}")
                continue
            
            # Generate audio for each chunk and combine
            combined_audio = AudioSegment.silent(duration=500)  # Start with silence
            
            for j, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(f"   Processing chunk {j+1}/{len(chunks)}...")
                
                # Create temporary file for this chunk
                temp_wav = output_dir / f"temp_{chapter_name}_chunk_{j}.wav"
                
                try:
                    # Generate audio using direct XTTS
                    result = xtts.synthesize_audio(
                        text=chunk,
                        output_path=str(temp_wav),
                        speaker_wav=speaker_wav,
                        language="en"
                    )
                    
                    if result and temp_wav.exists():
                        # Load and add to combined audio
                        chunk_audio = AudioSegment.from_wav(str(temp_wav))
                        combined_audio += chunk_audio + AudioSegment.silent(duration=300)
                        
                        # Clean up temp file
                        temp_wav.unlink()
                    else:
                        if progress_callback:
                            progress_callback(f"⚠️ Failed to generate chunk {j+1}")
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ Error with chunk {j+1}: {e}")
                    # Clean up temp file if it exists
                    if temp_wav.exists():
                        temp_wav.unlink()
            
            # Export combined audio as MP3
            if len(combined_audio) > 1000:  # Only save if we have substantial audio
                combined_audio.export(str(mp3_file), format="mp3", bitrate="192k")
                created_files.append(mp3_file)
                
                if progress_callback:
                    progress_callback(f"✅ Saved: {mp3_file.name} ({len(combined_audio)/1000:.1f}s)")
            else:
                if progress_callback:
                    progress_callback(f"⚠️ No audio generated for: {chapter_name}")
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {chapter_name}: {e}")
            continue
    
    if progress_callback:
        progress_callback(f"🎉 Direct XTTS conversion complete! Created {len(created_files)} files.")
    
    return created_files


def convert_txt_to_mp3(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
    voice: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    cloned_voices_data: Optional[dict] = None,
    xtts_settings: Optional[dict] = None,
):
    """
    Convert all .txt files in input_dir to MP3 using Mozilla TTS and save to output_dir.
    Handles both standard TTS models and cloned voices.
    DEPRECATED: Use enhanced_tts.convert_txt_to_mp3_enhanced() for better quality.
    """
    
    # Decide whether to use XTTS paths (only for xtts_v2 or cloned voices)
    is_xtts_requested = False
    model_lower = (model_name or "").lower()
    try:
        from voice_cloning_fix import is_cloned_voice as _is_cloned_voice
        is_xtts_requested = _is_cloned_voice(model_name) or ("xtts" in model_lower)
    except Exception:
        is_xtts_requested = (model_lower.startswith("cloned_") or ("xtts" in model_lower))
    
    # If NOT using XTTS, skip XTTS paths and go straight to enhanced/basic engines
    if not is_xtts_requested:
        try:
            from enhanced_tts import convert_txt_to_mp3_enhanced
            if progress_callback:
                progress_callback("🔄 Using enhanced TTS engine (non-XTTS model)...")
            created_files = convert_txt_to_mp3_enhanced(
                input_dir=input_dir,
                output_dir=output_dir,
                model_key="vctk_multi" if "vctk" in model_lower else None,
                progress_callback=progress_callback
            )
            if created_files:
                return created_files
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Enhanced TTS not available or failed ({e}), using basic TTS...")
        # Fall back to basic path below (original implementation)
    
    # First try the final working XTTS implementation (more likely to produce real speech)
    if XTTS_WORKING:
        try:
            if progress_callback:
                progress_callback("🔄 Using final working XTTS implementation...")
            
            created_files = convert_with_final_xtts(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name=model_name,
                progress_callback=progress_callback,
                cloned_voices_data=cloned_voices_data,
                xtts_settings=xtts_settings,
            )
            
            if created_files:
                return created_files
            else:
                if progress_callback:
                    progress_callback("⚠️ Final XTTS created no files, trying Direct XTTS as fallback...")
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Final XTTS failed ({e}), trying Direct XTTS...")
    
    # Second try the Direct XTTS implementation (may fall back to placeholder if inference fails)
    if DIRECT_XTTS_AVAILABLE:
        try:
            if progress_callback:
                progress_callback("🔄 Using Direct XTTS implementation (fallback)...")
            
            created_files = convert_with_direct_xtts(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name=model_name,
                progress_callback=progress_callback,
                cloned_voices_data=cloned_voices_data
            )
            
            if created_files:
                return created_files
            else:
                if progress_callback:
                    progress_callback("⚠️ Direct XTTS created no files, trying enhanced TTS...")
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Direct XTTS failed ({e}), trying enhanced TTS...")
    
    # Try to use enhanced TTS as fallback
    try:
        from enhanced_tts import convert_txt_to_mp3_enhanced
        if progress_callback:
            progress_callback("🔄 Using enhanced TTS engine for better quality...")
        created_files = convert_txt_to_mp3_enhanced(
            input_dir=input_dir,
            output_dir=output_dir,
            model_key="vctk_multi",  # Use VCTK multi-speaker as default
            progress_callback=progress_callback
        )
        if created_files:  # Only return if files were actually created
            return created_files
        else:
            if progress_callback:
                progress_callback("⚠️ Enhanced TTS created no files, falling back to basic TTS...")
    except ImportError:
        if progress_callback:
            progress_callback("⚠️ Enhanced TTS not available, using basic TTS...")
    except Exception as e:
        if progress_callback:
            progress_callback(f"⚠️ Enhanced TTS failed ({e}), falling back to basic TTS...")
    
    # Fallback to original implementation
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_wav = None  # For cloned voices
    
    # Handle cloned voices differently
    if model_name.startswith('cloned_'):
        # Extract the voice name from the model_name
        voice_name = model_name.replace('cloned_', '').replace('_', ' ').title()
        
        # Try to get speaker wav file from cloned voices data
        if cloned_voices_data and voice_name in cloned_voices_data:
            speaker_wav = cloned_voices_data[voice_name].get('speaker_wav')
        
        # For cloned voices, try XTTS-v2 first, then YourTTS
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if progress_callback:
                progress_callback(f"🎬 Using XTTS-v2 for cloned voice: {voice_name}")
        except Exception:
            try:
                tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
                if progress_callback:
                    progress_callback(f"🎬 Using YourTTS for cloned voice: {voice_name}")
            except Exception as e:
                # Fallback to VCTK multi-speaker
                if progress_callback:
                    progress_callback(f"⚠️ Cloned voice models not available, using VCTK multi-speaker")
                tts = TTS("tts_models/en/vctk/vits")
                speaker_wav = None  # Clear speaker_wav since we're using fallback
    else:
        # Standard TTS model - prefer higher quality models
        model_priority = [
            model_name,  # User's choice first
            "tts_models/en/vctk/vits",  # Multi-speaker VCTK
            "tts_models/en/ljspeech/glow-tts",  # GlowTTS
            "tts_models/en/ljspeech/tacotron2-DDC"  # Fallback
        ]
        
        tts = None
        for model in model_priority:
            try:
                tts = TTS(model)
                if progress_callback:
                    progress_callback(f"🎤 Using TTS model: {model}")
                break
            except Exception as e:
                if progress_callback and model == model_name:
                    progress_callback(f"⚠️ Model {model} not available: {e}")
                continue
        
        if not tts:
            raise RuntimeError("No TTS models available")
    
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    for txt_file in txt_files:
        chapter_name = txt_file.stem
        try:
            if progress_callback:
                progress_callback(f"🔊 Converting: {chapter_name}")

            text = txt_file.read_text(encoding="utf-8").strip()
            text = normalize_text(text)
            chunks = split_into_chunks(text)

            if not chunks:
                raise ValueError("No valid text chunks found for synthesis.")

            mp3_file = output_dir / f"{chapter_name}.mp3"
            
            # Determine if we need a speaker ID for multi-speaker models
            speaker_id = None
            model_name_lower = model_name.lower()
            if 'vctk' in model_name_lower or 'multi' in model_name_lower:
                speaker_id = 'p225'  # Default to a reliable VCTK speaker
            
            synthesize_chunks_to_mp3(chunks, mp3_file, tts, speaker_wav, speaker_id)

            if progress_callback:
                progress_callback(f"✅ Saved: {mp3_file.name}")

        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error with {chapter_name}: {str(e)}")
            continue

    if progress_callback:
        progress_callback("🎉 All chapters converted to MP3.")
