"""Ito: natural-sounding streaming text-to-speech small enough for an ESP32-S3.

    from ito import Ito
    wav = Ito.load().synthesize("Good morning! The coffee is ready.")   # float32, 24 kHz
"""
__version__ = "3.0.0"
__all__ = ["Ito", "Chain", "Front", "Vocoder"]


def __getattr__(name):            # lazy: `import ito.text` must work without PyTorch (esp32/tools/say.py)
    if name in ("Ito", "Chain"):
        from . import synth
        return getattr(synth, name)
    if name == "Front":
        from .front import Front
        return Front
    if name == "Vocoder":
        from .vocoder import Vocoder
        return Vocoder
    raise AttributeError(name)
