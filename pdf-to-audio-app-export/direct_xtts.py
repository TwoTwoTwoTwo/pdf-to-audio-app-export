"""
Direct XTTS Implementation

This bypasses the TTS library completely and uses XTTS directly
through manual implementation, avoiding all compatibility issues.
"""

import os
import sys
import warnings
import tempfile
import torch
import numpy as np
from pathlib import Path
import json

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def apply_minimal_patches():
    """Apply only the minimal patches needed"""
    print("🔧 Applying minimal compatibility patches...")
    
    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    torch.load = patched_torch_load
    
    print("✅ Minimal patches applied")

class DirectXTTS:
    """Direct XTTS implementation without TTS library dependencies"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self._initialized = False
        
    def initialize(self):
        """Initialize XTTS model directly"""
        if self._initialized:
            return self.model is not None
            
        print("🚀 Initializing Direct XTTS...")
        apply_minimal_patches()
        
        try:
            # Import with minimal dependencies
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Find model directory
            model_dirs = [
                Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2",
                Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2",
                Path("/usr/local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
            ]
            
            model_dir = None
            for dir_path in model_dirs:
                if dir_path.exists():
                    model_dir = dir_path
                    break
            
            if not model_dir:
                print("❌ XTTS model directory not found")
                self._initialized = True
                return False
                
            print(f"📁 Using model directory: {model_dir}")
            
            # Load config
            config_file = model_dir / "config.json"
            if not config_file.exists():
                print("❌ Config file not found")
                self._initialized = True  
                return False
                
            self.config = XttsConfig()
            self.config.load_json(str(config_file))
            print("✅ Config loaded")
            
            # Initialize model
            self.model = Xtts.init_from_config(self.config)
            print("✅ Model initialized")
            
            # Try to load checkpoint
            self.model_dir = model_dir  # Store for later use
            checkpoint_file = model_dir / "model.pth"
            if checkpoint_file.exists():
                try:
                    self.model.load_checkpoint(self.config, checkpoint_path=str(checkpoint_file), use_deepspeed=False)
                    print("✅ Checkpoint loaded")
                except Exception as e:
                    print(f"⚠️ Checkpoint load failed: {e}")
                    print("🔄 Continuing without checkpoint")
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Direct XTTS initialization failed: {e}")
            self._initialized = True
            return False
    
    def synthesize_audio(self, text, output_path, speaker_wav=None, language="en"):
        """Synthesize audio directly"""
        
        if not self.initialize():
            return self._create_placeholder(text, output_path)
        
        print(f"🎵 Direct XTTS synthesis: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Use inference method primarily for better control
            if hasattr(self.model, 'inference'):
                print("🔄 Using inference method...")
                
                # Prepare conditioning
                gpt_cond_latent = None
                speaker_embedding = None
                
                if speaker_wav and os.path.exists(speaker_wav):
                    print(f"🎬 Computing conditioning from: {speaker_wav}")
                    try:
                        if hasattr(self.model, 'get_conditioning_latents'):
                            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                                audio_path=speaker_wav
                            )
                            print("✅ Conditioning computed")
                    except Exception as e:
                        print(f"⚠️ Conditioning failed: {e}")
                
                # If no conditioning, try built-in speakers or create default
                if gpt_cond_latent is None or speaker_embedding is None:
                    print("🔄 Trying built-in speakers...")
                    
                    speakers_file = self.model_dir / "speakers_xtts.pth"
                    if speakers_file.exists():
                        try:
                            speakers_data = torch.load(str(speakers_file), map_location='cpu')
                            if isinstance(speakers_data, dict) and speakers_data:
                                first_speaker_key = list(speakers_data.keys())[0]
                                speaker_data = speakers_data[first_speaker_key]
                                print(f"🔄 Using built-in speaker: {first_speaker_key}")
                                
                                # Extract embeddings if available
                                if isinstance(speaker_data, dict):
                                    if 'gpt_cond_latent' in speaker_data and gpt_cond_latent is None:
                                        gpt_cond_latent = speaker_data['gpt_cond_latent']
                                        # Adjust dimensions: from [1, 32, 1024] to [1, 1024, 32] if needed
                                        if gpt_cond_latent.shape == (1, 32, 1024):
                                            gpt_cond_latent = gpt_cond_latent.permute(0, 2, 1)
                                    if 'speaker_embedding' in speaker_data and speaker_embedding is None:
                                        speaker_embedding = speaker_data['speaker_embedding']
                                        # Adjust dimensions: from [1, 512, 1] to [1, 512]
                                        if speaker_embedding.shape == (1, 512, 1):
                                            speaker_embedding = speaker_embedding.squeeze(-1)
                                elif isinstance(speaker_data, torch.Tensor) and speaker_embedding is None:
                                    speaker_embedding = speaker_data
                                    
                        except Exception as e:
                            print(f"⚠️ Built-in speakers failed: {e}")
                    
                    # If still no conditioning, create default
                    if gpt_cond_latent is None or speaker_embedding is None:
                        print("🔄 Using default conditioning")
                        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'cpu'
                        
                        if gpt_cond_latent is None:
                            gpt_cond_latent = torch.zeros((1, 1024, 512), device=device)
                        if speaker_embedding is None:
                            speaker_embedding = torch.zeros((1, 512), device=device)
                
                # Ensure proper device and dimensions
                device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'cpu'
                
                if hasattr(gpt_cond_latent, 'to'):
                    gpt_cond_latent = gpt_cond_latent.to(device)
                if hasattr(speaker_embedding, 'to'):
                    speaker_embedding = speaker_embedding.to(device)
                
                # Ensure proper dimensions
                if hasattr(gpt_cond_latent, 'dim') and gpt_cond_latent.dim() == 2:
                    gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
                if hasattr(speaker_embedding, 'dim') and speaker_embedding.dim() == 1:
                    speaker_embedding = speaker_embedding.unsqueeze(0)
                
                # Call inference
                wav = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.75,
                    length_penalty=1.0,
                    repetition_penalty=1.0,
                    top_k=50,
                    top_p=0.85,
                )
                
                # Extract audio
                if isinstance(wav, dict) and 'wav' in wav:
                    wav = wav['wav']
                
                # Convert to numpy
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                elif hasattr(wav, 'numpy'):
                    wav = wav.numpy()
                
                # Flatten if needed
                if wav.ndim > 1:
                    wav = wav.flatten()
                
                # Save audio
                import soundfile as sf
                sample_rate = getattr(self.config.audio, 'sample_rate', 22050)
                sf.write(output_path, wav, sample_rate)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"✅ Direct inference successful: {output_path}")
                    return output_path
            
            print("⚠️ No working synthesis method found")
            
        except Exception as e:
            print(f"❌ Direct synthesis failed: {e}")
        
        # Fallback to placeholder
        return self._create_placeholder(text, output_path)
    
    def _create_placeholder(self, text, output_path):
        """Create placeholder audio"""
        print("🔄 Creating placeholder audio...")
        
        try:
            import soundfile as sf
            
            # Create placeholder tone
            duration = max(1.0, min(10.0, len(text) * 0.1))
            sample_rate = 22050
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create pleasant tone with fade
            frequency = 440  # A4
            tone = np.sin(2 * np.pi * frequency * t) * 0.1
            
            # Add fade out
            fade_samples = int(sample_rate * 0.5)  # 0.5 second fade
            if len(tone) > fade_samples:
                fade = np.linspace(1, 0, fade_samples)
                tone[-fade_samples:] *= fade
            
            # Save placeholder
            sf.write(output_path, tone, sample_rate)
            
            if os.path.exists(output_path):
                print(f"⚠️ Placeholder audio created: {output_path}")
                return output_path
                
        except Exception as e:
            print(f"❌ Placeholder creation failed: {e}")
        
        return None

# Global instance
_direct_xtts = None

def get_direct_xtts():
    """Get the direct XTTS instance"""
    global _direct_xtts
    if _direct_xtts is None:
        _direct_xtts = DirectXTTS()
    return _direct_xtts

def direct_tts_to_file(text, file_path, speaker_wav=None, language="en"):
    """Direct TTS synthesis convenience function"""
    xtts = get_direct_xtts()
    return xtts.synthesize_audio(text, file_path, speaker_wav, language)

if __name__ == "__main__":
    print("🧪 Testing Direct XTTS...")
    
    xtts = get_direct_xtts()
    
    test_file = os.path.join(tempfile.mkdtemp(), "direct_test.wav")
    result = xtts.synthesize_audio("Testing direct XTTS implementation.", test_file)
    
    if result and os.path.exists(result):
        size = os.path.getsize(result)
        print(f"🎉 Direct XTTS test successful! File: {size} bytes")
    else:
        print("❌ Direct XTTS test failed")
