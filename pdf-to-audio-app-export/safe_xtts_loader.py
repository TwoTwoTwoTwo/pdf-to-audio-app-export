#!/usr/bin/env python3
"""
Safe XTTS-v2 loader with comprehensive model validation and initialization checks.
Ensures proper model initialization and prevents NoneType attribute errors.
"""

# Install import hook early so XTTS modules are patched at import time
try:
    import import_hook_xtts_patch  # installs hook on import
except Exception:
    pass

import torch
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
import importlib
import sys

class SafeXTTSLoader:
    """Safe loader for XTTS-v2 with comprehensive validation"""
    
    def __init__(self):
        self.model_cache = {}
        self.apply_compatibility_fixes()
    
    def apply_compatibility_fixes(self):
        """Apply necessary compatibility fixes for XTTS-v2"""
        try:
            # Apply the final XTTS fix (most comprehensive)
            from final_xtts_fix import apply_final_xtts_fix
            apply_final_xtts_fix()
            print("✅ Applied final XTTS fix")
        except ImportError:
            try:
                # Fallback to comprehensive fix
                from xtts_transformers_fix import apply_comprehensive_xtts_transformers_fix
                apply_comprehensive_xtts_transformers_fix()
                print("✅ Applied comprehensive XTTS Transformers 4.50+ fix")
            except ImportError:
                try:
                    # Fallback to surgical fix
                    from xtts_surgical_fix import apply_surgical_fix
                    apply_surgical_fix()
                    print("✅ Applied surgical XTTS fix")
                except ImportError:
                    # Basic torch.load fix as last resort
                    if not hasattr(torch, '_safe_xtts_patched'):
                        original_load = torch.load
                        def safe_torch_load(*args, **kwargs):
                            if 'weights_only' not in kwargs:
                                kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = safe_torch_load
                        torch._safe_xtts_patched = True
                        print("✅ Applied basic torch.load fix")
    
    def validate_model_components(self, tts_model) -> bool:
        """Validate that the TTS object looks usable without assuming internal structure."""
        if tts_model is None:
            return False

        # If the object has a tts_to_file method, consider it usable
        if hasattr(tts_model, 'tts_to_file') and callable(getattr(tts_model, 'tts_to_file')):
            # If it also exposes .model, perform extra checks for XTTS
            if hasattr(tts_model, 'model') and tts_model.model is not None:
                # Check for XTTS-specific components if present
                if hasattr(tts_model.model, 'gpt') and tts_model.model.gpt is not None:
                    if not hasattr(tts_model.model.gpt, '_from_model_config'):
                        print("⚠️ GPT component missing _from_model_config - attempting to patch")
                        def dummy_from_model_config(cls, config):
                            return cls(**config)
                        tts_model.model.gpt.__class__._from_model_config = classmethod(dummy_from_model_config)
                if hasattr(tts_model.model, 'hifigan_decoder') and tts_model.model.hifigan_decoder is None:
                    print("❌ XTTS HiFiGAN decoder is None")
                    return False
            print("✅ TTS object validated (tts_to_file available)")
            return True

        # Fallback: if it has a tts method, still usable
        if hasattr(tts_model, 'tts') and callable(getattr(tts_model, 'tts')):
            print("✅ TTS object validated (tts available)")
            return True

        print("❌ TTS object missing synthesis methods")
        return False
    
    def safe_load_xtts(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", gpu: bool = False) -> Optional[Any]:
        """
        Safely load XTTS-v2 model with comprehensive validation
        
        Args:
            model_name: The TTS model name to load
            gpu: Whether to use GPU (default: False for compatibility)
            
        Returns:
            Properly initialized TTS model or None if failed
        """
        cache_key = f"{model_name}_{gpu}"
        
        # Return cached model if available and valid
        if cache_key in self.model_cache:
            cached_model = self.model_cache[cache_key]
            if self.validate_model_components(cached_model):
                print(f"✅ Using cached XTTS model: {model_name}")
                return cached_model
            else:
                print("⚠️ Cached model validation failed, reloading...")
                del self.model_cache[cache_key]
        
        print(f"🔄 Loading XTTS model: {model_name}")
        
        # Clear any existing model cache
        if torch.cuda.is_available() and gpu:
            torch.cuda.empty_cache()
        
        # Multiple initialization strategies
        init_strategies = [
            # Strategy 1: Standard initialization
            lambda: self._init_standard(model_name, gpu),
            # Strategy 2: With explicit CPU device
            lambda: self._init_with_device(model_name, "cpu"),
            # Strategy 3: Force model re-download
            lambda: self._init_force_download(model_name, gpu),
            # Strategy 4: Manual model loading
            lambda: self._init_manual_loading(model_name, gpu)
        ]
        
        for i, strategy in enumerate(init_strategies, 1):
            try:
                print(f"🔄 Trying initialization strategy {i}...")
                tts_model = strategy()
                
                if tts_model is not None and self.validate_model_components(tts_model):
                    print(f"✅ XTTS model loaded successfully (strategy {i})")
                    self.model_cache[cache_key] = tts_model
                    return tts_model
                else:
                    print(f"⚠️ Strategy {i} failed validation")
                    
            except Exception as e:
                print(f"⚠️ Strategy {i} failed: {str(e)[:100]}...")
                continue
        
        print("❌ All XTTS initialization strategies failed")
        return None
    
    def _init_standard(self, model_name: str, gpu: bool):
        """Standard TTS initialization"""
        from TTS.api import TTS
        return TTS(model_name=model_name, progress_bar=False, gpu=gpu)
    
    def _init_with_device(self, model_name: str, device: str):
        """Initialize with explicit device specification"""
        from TTS.api import TTS
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        return tts.to(device)
    
    def _init_force_download(self, model_name: str, gpu: bool):
        """Force model re-download and initialization"""
        from TTS.api import TTS
        import shutil
        
        # Clear model cache directory
        cache_dir = Path.home() / ".cache" / "tts"
        if cache_dir.exists():
            model_cache_dirs = list(cache_dir.glob("*xtts*"))
            for cache_dir in model_cache_dirs:
                try:
                    shutil.rmtree(cache_dir)
                    print(f"🗑️ Cleared cache: {cache_dir}")
                except:
                    pass
        
        return TTS(model_name=model_name, progress_bar=False, gpu=gpu)
    
    def _init_manual_loading(self, model_name: str, gpu: bool):
        """Manual model loading with step-by-step validation"""
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        
        # Use model manager for more controlled loading
        manager = ModelManager()
        model_path, config_path, model_item = manager.download_model(model_name)
        
        print(f"📁 Model path: {model_path}")
        print(f"⚙️ Config path: {config_path}")
        
        # Initialize with downloaded paths
        tts = TTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=gpu)
        
        return tts
    
    def safe_generate_audio(self, tts_model, text: str, speaker_wav: str = None, language: str = "en", output_path: str = None) -> Optional[str]:
        """
        Safely generate audio with comprehensive pre-generation validation
        
        Args:
            tts_model: The TTS model to use
            text: Text to synthesize
            speaker_wav: Path to speaker reference audio (for cloning)
            language: Language code
            output_path: Output file path (optional)
            
        Returns:
            Path to generated audio file or None if failed
        """
        # Pre-generation validation
        if tts_model is None:
            print("❌ TTS model is None")
            return None
            
        if not self.validate_model_components(tts_model):
            print("❌ Model validation failed before generation")
            return None
            
        if not text or len(text.strip()) == 0:
            print("❌ Empty text provided")
            return None
            
        # Check speaker wav file if provided
        if speaker_wav:
            if not os.path.exists(speaker_wav):
                print(f"❌ Speaker WAV file not found: {speaker_wav}")
                return None
                
            # Validate audio file
            try:
                import librosa
                y, sr = librosa.load(speaker_wav, sr=None)
                duration = len(y) / sr
                print(f"✅ Speaker audio: {duration:.2f}s, {sr}Hz")
                
                if duration < 1.0:
                    print("⚠️ Speaker audio is very short, may affect quality")
                    
            except Exception as e:
                print(f"⚠️ Could not validate speaker audio: {e}")
        
        # Prepare generation arguments
        if output_path is None:
            import tempfile
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        
        try:
            print(f"🎯 Generating audio: {len(text)} characters")
            
            # Build kwargs based on model type and available parameters
            # XTTS-v2 is multilingual and requires `language`, so always include it.
            tts_kwargs = {"text": text, "language": language or "en"}
            
            # Prefer passing speaker_wav; if unsupported, we'll retry without it
            if speaker_wav:
                tts_kwargs["speaker_wav"] = speaker_wav
                print(f"🎤 Attempting voice cloning mode with {tts_kwargs['language']}")
            
            # Generate audio with tolerant argument handling
            try:
                tts_model.tts_to_file(file_path=output_path, **tts_kwargs)
            except TypeError as te:
                # If signature mismatch, retry conservatively by removing only speaker_wav
                if 'speaker_wav' in tts_kwargs:
                    print("🔄 Retrying generation without speaker_wav due to signature mismatch")
                    tts_kwargs2 = {"text": text, "language": language or "en"}
                    tts_model.tts_to_file(file_path=output_path, **tts_kwargs2)
                else:
                    raise
            
            # Validate output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:  # > 1KB
                print(f"✅ Audio generated successfully: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                print("❌ Generated audio file is empty or invalid")
                return None
                
        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            
            # Enhanced error diagnostics
            import traceback
            tb_str = traceback.format_exc()
            
            if "_from_model_config" in str(e):
                print("❌ _from_model_config error detected - model component is None")
                print("💡 This usually means the GPT component failed to initialize")
                
            if "NoneType" in str(e):
                print("❌ NoneType error detected - checking model state...")
                self._diagnose_model_state(tts_model)
                
            return None
    
    def _diagnose_model_state(self, tts_model):
        """Diagnose the current state of the TTS model"""
        print("🔍 Model State Diagnosis:")
        
        if tts_model is None:
            print("   - TTS model: None")
            return
            
        print(f"   - TTS model type: {type(tts_model)}")
        print(f"   - Has model attr: {hasattr(tts_model, 'model')}")
        
        if hasattr(tts_model, 'model'):
            print(f"   - Model value: {tts_model.model}")
            print(f"   - Model type: {type(tts_model.model) if tts_model.model else 'None'}")
            
            if tts_model.model:
                print(f"   - Has GPT: {hasattr(tts_model.model, 'gpt')}")
                if hasattr(tts_model.model, 'gpt'):
                    print(f"   - GPT value: {tts_model.model.gpt}")
                    if tts_model.model.gpt:
                        print(f"   - GPT type: {type(tts_model.model.gpt)}")
                        print(f"   - Has _from_model_config: {hasattr(tts_model.model.gpt, '_from_model_config')}")

# Global instance for easy access
_safe_loader = SafeXTTSLoader()

def safe_load_xtts_validated(model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", gpu: bool = False):
    """
    Global function to safely load XTTS with validation
    """
    return _safe_loader.safe_load_xtts(model_name, gpu)

def safe_generate_xtts_audio(tts_model, text: str, speaker_wav: str = None, language: str = "en", output_path: str = None):
    """
    Global function to safely generate audio with validation
    """
    return _safe_loader.safe_generate_audio(tts_model, text, speaker_wav, language, output_path)

def validate_xtts_model(tts_model) -> bool:
    """
    Global function to validate XTTS model components
    """
    return _safe_loader.validate_model_components(tts_model)
