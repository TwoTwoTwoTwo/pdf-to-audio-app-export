#!/usr/bin/env python3
"""
XTTS import hook patch

Patches GPT2InferenceModel to inherit from GenerationMixin at import time,
ensuring Transformers 4.50+ compatibility before any XTTS instances are created.

- Hooks into import of TTS.tts.models.xtts
- When module loads, modifies GPT2InferenceModel bases to include GenerationMixin
- Adds minimal _from_model_config and generate methods if missing

Safe to import multiple times; it installs the hook only once.
"""
import sys
import importlib
import importlib.abc
import importlib.util
import types

_installed = False

class _XTTSImportFinder(importlib.abc.MetaPathFinder):
    target_module = "TTS.tts.models.xtts"

    def find_spec(self, fullname, path, target=None):
        if fullname == self.target_module:
            # Use default mechanism but wrap loader
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception:
                spec = None
            if spec and spec.loader:
                spec.loader = _XTTSImportLoader(spec.loader)
            return spec
        return None

class _XTTSImportLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None  # use default module creation

    def exec_module(self, module):
        # First, execute the original module
        self._wrapped.exec_module(module)
        # Then patch
        try:
            _patch_xtts_module(module)
        except Exception as e:
            print(f"⚠️ XTTS import patch failed: {e}")


def _patch_xtts_module(module: types.ModuleType):
    """Patch GPT2InferenceModel inside the imported XTTS module."""
    try:
        # Resolve class
        GPT2InferenceModel = getattr(module, "GPT2InferenceModel", None)
        if GPT2InferenceModel is None:
            # Some versions may have nested paths
            return

        # Import GenerationMixin lazily
        try:
            from transformers.generation.utils import GenerationMixin
        except Exception:
            try:
                from transformers import GenerationMixin  # alt path
            except Exception as e:
                print(f"⚠️ Could not import GenerationMixin: {e}")
                return

        # If already a subclass, nothing to do
        if issubclass(GPT2InferenceModel, GenerationMixin):
            return

        # Modify bases to include GenerationMixin
        try:
            GPT2InferenceModel.__bases__ = GPT2InferenceModel.__bases__ + (GenerationMixin,)
        except TypeError:
            # Some environments prohibit modifying __bases__ directly; define dynamic subclass
            name = GPT2InferenceModel.__name__
            Patched = type(name, (GPT2InferenceModel, GenerationMixin), {})
            setattr(module, name, Patched)
            GPT2InferenceModel = Patched

        # Ensure _from_model_config exists
        if not hasattr(GPT2InferenceModel, "_from_model_config"):
            @classmethod
            def _from_model_config(cls, config, **kwargs):
                return cls(config=config, **kwargs) if "config" in cls.__init__.__code__.co_varnames else cls(**kwargs)
            setattr(GPT2InferenceModel, "_from_model_config", _from_model_config)

        # Ensure a basic generate method is available (some XTTS code may call it)
        if not hasattr(GPT2InferenceModel, "generate"):
            def generate(self, *args, **kwargs):
                if hasattr(super(GPT2InferenceModel, self), "generate"):
                    return super(GPT2InferenceModel, self).generate(*args, **kwargs)
                # Fallback to forward if generate is not present upstream
                return self.forward(*args, **kwargs)
            setattr(GPT2InferenceModel, "generate", generate)

        print("✅ XTTS GPT2InferenceModel patched with GenerationMixin at import time")
    except Exception as e:
        print(f"⚠️ Failed to patch XTTS module: {e}")


def install_xtts_import_hook():
    global _installed
    if _installed:
        return True
    try:
        # If already imported, patch immediately
        if "TTS.tts.models.xtts" in sys.modules:
            try:
                _patch_xtts_module(sys.modules["TTS.tts.models.xtts"])
            except Exception:
                pass
        # Install meta_path finder at front so it triggers before standard import
        finder = _XTTSImportFinder()
        sys.meta_path.insert(0, finder)
        _installed = True
        print("🔧 XTTS import hook installed")
        return True
    except Exception as e:
        print(f"❌ Could not install XTTS import hook: {e}")
        return False

# Auto-install on import for convenience
try:
    install_xtts_import_hook()
except Exception:
    pass

