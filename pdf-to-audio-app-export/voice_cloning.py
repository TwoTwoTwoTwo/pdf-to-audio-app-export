#!/usr/bin/env python3
"""
Voice Cloning Module
Extracts voice models from YouTube videos for TTS voice cloning
"""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil
import torch

# Fix PyTorch 2.6+ compatibility for TTS models
def fix_pytorch_tts_compatibility():
    """Fix PyTorch compatibility issues with TTS model loading"""
    try:
        # Add safe globals for TTS config classes
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig])
        
        # Add other TTS classes that might be needed
        try:
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.configs.tacotron2_config import Tacotron2Config
            torch.serialization.add_safe_globals([XttsAudioConfig, XttsArgs, VitsConfig, Tacotron2Config])
        except ImportError:
            pass  # These might not exist in all TTS versions
            
    except Exception as e:
        # If we can't add safe globals, patch torch.load instead
        if not hasattr(torch, '_original_load_patched_voice_cloning'):
            original_load = torch.load
            def patched_load(*args, **kwargs):
                # For TTS models, disable weights_only to avoid loading errors
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            torch._original_load_patched_voice_cloning = True

# Apply the fix on import
fix_pytorch_tts_compatibility()

try:
    import yt_dlp
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

class VoiceCloner:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="voice_clone_"))
        self.sample_rate = 22050  # Standard for TTS models
        self.min_clip_length = 2.0  # Minimum seconds for individual segments
        self.max_clip_length = 10.0  # Maximum seconds for individual segments
        self.target_combined_length = 60.0  # Target 1 minute for XTTS-v2 voice cloning
        
    def __del__(self):
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def extract_audio_from_youtube(self, url: str, output_path: Path) -> bool:
        """Extract audio from YouTube video"""
        
        if not YOUTUBE_AVAILABLE:
            raise ImportError("yt-dlp is required for YouTube audio extraction. Install with: pip install yt-dlp")
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': [
                    '-ar', str(self.sample_rate)
                ],
                'prefer_ffmpeg': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'unknown')
                duration = info.get('duration', 0)
                
                print(f"📺 Video: {title}")
                print(f"⏱️ Duration: {duration//60}:{duration%60:02d}")
                
                # Check if video is too long (over 30 minutes)
                if duration > 1800:
                    print("⚠️ Warning: Video is longer than 30 minutes. This may take a while and use significant disk space.")
                
                # Download audio
                ydl.download([url])
                
                return True
                
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return False
    
    def clean_audio_for_voice_cloning(self, audio_path: Path) -> List[Path]:
        """Clean and split audio into voice samples suitable for cloning"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            raise ImportError("pydub is required for audio processing. Install with: pip install pydub")
        
        print("🧹 Processing audio for voice cloning...")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Remove silence and split into segments
            print("🔪 Splitting on silence...")
            segments = split_on_silence(
                audio,
                min_silence_len=500,  # 0.5 seconds of silence
                silence_thresh=-40,   # dB
                keep_silence=100      # Keep 0.1 seconds of silence
            )
            
            # Filter segments by length
            good_segments = []
            for i, segment in enumerate(segments):
                duration = len(segment) / 1000.0  # Convert to seconds
                if self.min_clip_length <= duration <= self.max_clip_length:
                    good_segments.append((i, segment))
            
            print(f"✅ Found {len(good_segments)} suitable voice segments")
            
            # Save the best segments
            output_files = []
            clips_dir = self.temp_dir / "voice_clips"
            clips_dir.mkdir(exist_ok=True)
            
            # Take up to 20 best segments (or all if fewer)
            selected_segments = good_segments[:20]
            
            for i, (orig_idx, segment) in enumerate(selected_segments):
                clip_path = clips_dir / f"clip_{i+1:02d}.wav"
                segment.export(clip_path, format="wav")
                output_files.append(clip_path)
                
                duration = len(segment) / 1000.0
                print(f"  💾 Saved clip {i+1}: {duration:.1f}s")
            
            return output_files
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return []
    
    def create_voice_profile(self, voice_clips: List[Path], speaker_name: str) -> Dict:
        """Create a voice profile from audio clips"""
        
        if not voice_clips:
            raise ValueError("No voice clips provided")
        
        # Analyze voice characteristics
        print("🔍 Analyzing voice characteristics...")
        
        total_duration = 0
        clip_info = []
        
        for clip_path in voice_clips:
            if AUDIO_PROCESSING_AVAILABLE:
                try:
                    audio = AudioSegment.from_file(clip_path)
                    duration = len(audio) / 1000.0
                    total_duration += duration
                    
                    # Basic audio analysis
                    rms = audio.rms
                    max_amplitude = audio.max
                    
                    clip_info.append({
                        "file": clip_path.name,
                        "duration": duration,
                        "rms": rms,
                        "max_amplitude": max_amplitude
                    })
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not analyze {clip_path.name}: {e}")
        
        # Create voice profile
        voice_profile = {
            "speaker_name": speaker_name,
            "total_clips": len(voice_clips),
            "total_duration": round(total_duration, 2),
            "sample_rate": self.sample_rate,
            "clips": clip_info,
            "voice_characteristics": {
                "estimated_gender": "unknown",  # Would need more advanced analysis
                "estimated_age": "unknown",     # Would need more advanced analysis
                "voice_type": "cloned",
                "quality_score": min(len(voice_clips) / 10.0, 1.0)  # Based on number of samples
            },
            "tts_config": {
                "model_type": "voice_clone",
                "recommended_settings": {
                    "temperature": 0.7,
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0
                }
            }
        }
        
        return voice_profile
    
    def create_combined_sample_for_xtts(self, voice_clips: List[Path], speaker_name: str) -> Optional[Path]:
        """Combine voice clips into a longer sample for XTTS-v2 (target: 1 minute)"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            print("⚠️ Cannot create combined sample - pydub not available")
            return None
        
        if not voice_clips:
            print("⚠️ No voice clips to combine")
            return None
        
        try:
            # Load and combine clips
            combined_audio = AudioSegment.empty()
            current_duration = 0.0
            used_clips = []
            
            # Sort clips by duration (use longer clips first for better quality)
            sorted_clips = []
            for clip_path in voice_clips:
                try:
                    audio = AudioSegment.from_file(clip_path)
                    duration = len(audio) / 1000.0
                    sorted_clips.append((duration, clip_path, audio))
                except Exception as e:
                    print(f"⚠️ Could not load clip {clip_path.name}: {e}")
            
            # Sort by duration (longest first)
            sorted_clips.sort(key=lambda x: x[0], reverse=True)
            
            print(f"🎵 Combining clips to reach target {self.target_combined_length}s...")
            
            for duration, clip_path, audio in sorted_clips:
                if current_duration >= self.target_combined_length:
                    break
                
                # Add a small gap between clips (0.2 seconds)
                if len(used_clips) > 0:
                    silence = AudioSegment.silent(duration=200)  # 200ms
                    combined_audio += silence
                
                # Add the clip
                combined_audio += audio
                current_duration += duration + 0.2  # Include gap
                used_clips.append(clip_path.name)
                
                print(f"  ✅ Added {clip_path.name} ({duration:.1f}s) - Total: {current_duration:.1f}s")
            
            # If we don't have enough audio, repeat clips to reach target
            while current_duration < self.target_combined_length and sorted_clips:
                print(f"🔁 Repeating clips to reach target duration ({current_duration:.1f}s < {self.target_combined_length}s)")
                
                for duration, clip_path, audio in sorted_clips:
                    if current_duration >= self.target_combined_length:
                        break
                    
                    # Add gap
                    silence = AudioSegment.silent(duration=300)  # 300ms gap for repeated clips
                    combined_audio += silence
                    
                    # Add the clip
                    combined_audio += audio
                    current_duration += duration + 0.3
                    
                    print(f"  🔁 Repeated {clip_path.name} ({duration:.1f}s) - Total: {current_duration:.1f}s")
                
                # Safety break to avoid infinite loop
                if len(combined_audio) > self.target_combined_length * 1.5 * 1000:  # 1.5x target in ms
                    break
            
            # Save combined sample
            combined_path = self.temp_dir / f"combined_speaker_{speaker_name.lower().replace(' ', '_')}.wav"
            combined_audio.export(combined_path, format="wav")
            
            final_duration = len(combined_audio) / 1000.0
            print(f"✅ Created combined sample: {final_duration:.1f}s from {len(used_clips)} clips")
            
            # Provide feedback on duration
            if final_duration < 30.0:
                print("⚠️ Combined sample is shorter than 30 seconds - voice cloning quality may be reduced")
            elif final_duration >= 45.0:
                print("✅ Combined sample is good length for voice cloning (45+ seconds)")
            else:
                print("📏 Combined sample length is acceptable for voice cloning")
            
            return combined_path
            
        except Exception as e:
            print(f"❌ Error creating combined sample: {e}")
            return None
    
    def clone_voice_from_youtube(
        self, 
        youtube_url: str, 
        speaker_name: str,
        output_dir: Path
    ) -> Optional[Dict]:
        """Complete pipeline: YouTube URL -> Voice Clone"""
        
        print(f"🎬 Starting voice cloning from YouTube for '{speaker_name}'")
        print(f"📺 URL: {youtube_url}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract audio from YouTube
        print("\n📥 Step 1: Downloading audio from YouTube...")
        audio_success = self.extract_audio_from_youtube(youtube_url, self.temp_dir)
        if not audio_success:
            print("❌ Failed to extract audio from YouTube")
            return None
        
        # Find the downloaded audio file
        audio_files = list(self.temp_dir.glob("*.wav"))
        if not audio_files:
            print("❌ No audio file found after download")
            return None
        
        audio_file = audio_files[0]  # Use the first (should be only) file
        print(f"✅ Audio downloaded: {audio_file.name}")
        
        # Step 2: Clean and split audio
        print("\n🧹 Step 2: Processing audio for voice cloning...")
        voice_clips = self.clean_audio_for_voice_cloning(audio_file)
        
        if not voice_clips:
            print("❌ No suitable voice clips found")
            return None
        
        print(f"✅ Created {len(voice_clips)} voice clips")
        
        # Step 3: Create voice profile
        print("\n📊 Step 3: Creating voice profile...")
        voice_profile = self.create_voice_profile(voice_clips, speaker_name)
        
        # Step 4: Create combined long sample for XTTS-v2
        print("\n🎵 Step 4: Creating combined long sample for XTTS-v2...")
        combined_sample_path = self.create_combined_sample_for_xtts(voice_clips, speaker_name)
        
        # Step 5: Save voice model files
        print("\n💾 Step 5: Saving voice model...")
        
        # Create speaker directory
        speaker_dir = output_dir / f"voice_model_{speaker_name.lower().replace(' ', '_')}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Copy voice clips to output directory
        clips_output_dir = speaker_dir / "voice_samples"
        clips_output_dir.mkdir(exist_ok=True)
        
        for clip in voice_clips:
            shutil.copy2(clip, clips_output_dir)
        
        # Copy the combined sample for XTTS-v2
        if combined_sample_path and combined_sample_path.exists():
            combined_output_path = speaker_dir / "speaker_wav.wav"
            shutil.copy2(combined_sample_path, combined_output_path)
            print(f"✅ Main speaker sample for XTTS-v2: {combined_output_path}")
            voice_profile['speaker_wav'] = str(combined_output_path)
        
        # Save voice profile
        profile_path = speaker_dir / "voice_profile.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(voice_profile, f, indent=2, ensure_ascii=False)
        
        # Create usage instructions
        instructions = f"""# Voice Model: {speaker_name}

## Files:
- `voice_profile.json` - Voice characteristics and metadata
- `voice_samples/` - Directory containing {len(voice_clips)} voice sample clips

## Usage with Coqui TTS:

### 1. Install Coqui TTS with voice cloning support:
```bash
pip install TTS[voice_cloning]
```

### 2. Use voice cloning:
```python
from TTS.api import TTS

# Initialize TTS with voice cloning model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Clone voice using these samples
tts.tts_to_file(
    text="Your text here",
    file_path="output.wav",
    speaker_wav="{clips_output_dir}/clip_01.wav",  # Reference voice sample
    language="en"
)
```

### 3. For better results:
- Use multiple reference samples by concatenating them
- Ensure reference audio is clean and clear
- Text should be in the same language as the reference audio

## Voice Statistics:
- Total samples: {len(voice_clips)}
- Total duration: {voice_profile['total_duration']}s
- Quality score: {voice_profile['voice_characteristics']['quality_score']:.2f}

## Notes:
This voice model was created from: {youtube_url}
Generated on: {voice_profile.get('created_at', 'Unknown')}
"""
        
        instructions_path = speaker_dir / "README.md"
        instructions_path.write_text(instructions, encoding='utf-8')
        
        voice_profile['model_path'] = str(speaker_dir)
        voice_profile['created_at'] = str(Path().cwd())  # Placeholder
        
        print(f"✅ Voice model saved to: {speaker_dir}")
        print(f"📋 Usage instructions: {instructions_path}")
        
        return voice_profile
    
    def list_available_voices(self, voice_models_dir: Path) -> List[Dict]:
        """List all available voice models"""
        
        if not voice_models_dir.exists():
            return []
        
        voice_models = []
        
        for model_dir in voice_models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("voice_model_"):
                profile_path = model_dir / "voice_profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path, 'r', encoding='utf-8') as f:
                            profile = json.load(f)
                        
                        voice_models.append({
                            "name": profile.get("speaker_name", model_dir.name),
                            "path": str(model_dir),
                            "clips": profile.get("total_clips", 0),
                            "duration": profile.get("total_duration", 0),
                            "quality": profile.get("voice_characteristics", {}).get("quality_score", 0)
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load voice profile from {model_dir}: {e}")
        
        return sorted(voice_models, key=lambda x: x["quality"], reverse=True)

def create_voice_from_youtube_url(url: str, speaker_name: str, output_dir: str = "voice_models") -> bool:
    """Convenience function for creating voice model from YouTube URL"""
    
    cloner = VoiceCloner()
    
    try:
        result = cloner.clone_voice_from_youtube(url, speaker_name, Path(output_dir))
        return result is not None
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")
        return False

def main():
    """Command line interface for voice cloning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clone voices from YouTube videos for TTS")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("speaker_name", help="Name for the speaker/voice")
    parser.add_argument("--output", "-o", default="voice_models", help="Output directory for voice models")
    parser.add_argument("--list", action="store_true", help="List existing voice models")
    
    args = parser.parse_args()
    
    if args.list:
        voice_models = VoiceCloner().list_available_voices(Path(args.output))
        if voice_models:
            print("🎙️ Available Voice Models:")
            for model in voice_models:
                print(f"  • {model['name']} ({model['clips']} clips, {model['duration']:.1f}s, quality: {model['quality']:.2f})")
        else:
            print("No voice models found.")
        return 0
    
    # Check dependencies
    if not YOUTUBE_AVAILABLE:
        print("❌ yt-dlp is required. Install with: pip install yt-dlp")
        return 1
    
    if not AUDIO_PROCESSING_AVAILABLE:
        print("❌ pydub is required. Install with: pip install pydub")
        return 1
    
    # Clone voice
    success = create_voice_from_youtube_url(args.url, args.speaker_name, args.output)
    
    if success:
        print(f"\n🎉 Voice cloning completed successfully!")
        print(f"📁 Voice model saved in: {args.output}/voice_model_{args.speaker_name.lower().replace(' ', '_')}")
        return 0
    else:
        print("\n❌ Voice cloning failed")
        return 1

if __name__ == "__main__":
    exit(main())
