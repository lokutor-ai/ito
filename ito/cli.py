"""Command line: text -> WAV on the CPU (or a GPU).

    ito-tts "Good morning! The coffee is ready." -o out.wav
    ito-tts --voice g "Good morning! The coffee is ready." -o out.wav     # male voice
    ito-tts --phonemes "ɡʊd mˈɔːɹnɪŋ !" -o out.wav
    python -m ito "Long text. Several sentences are synthesized one by one." -o out.wav
"""
import argparse
import re
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ito-tts", description="Ito text-to-speech (24 kHz mono WAV)")
    ap.add_argument("text", nargs="?", help="English text")
    ap.add_argument("-o", "--out", default="ito.wav")
    ap.add_argument("--phonemes", default=None, help="StyleTTS 2-style phoneme string instead of text")
    ap.add_argument("--voice", default="d", type=str.lower, choices=("d", "g"),
                    help="d: female (LibriTTS-R 4970, default); g: male (LibriTTS-R 5105)")
    ap.add_argument("--model", default=None, help="checkpoint path, overrides --voice (default: ito_v3.pt or ito_v3_G.pt "
                                                  "from $ITO_WEIGHTS_DIR, models/, or Hugging Face)")
    ap.add_argument("--seed", type=int, default=0, help="source-noise seed")
    ap.add_argument("--chunk", type=int, default=8, help="frames per streamed chunk (8 = 100 ms)")
    ap.add_argument("--stream", action="store_true",
                    help="use the chunked streaming path (as on the chip; same audio, slower in Python)")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking rate (scales every duration by 1/speed)")
    ap.add_argument("--style", default="mean", choices=("mean", "predicted"),
                    help="mean: the fixed style the chip uses (default); predicted: text->style predictor")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if not a.text and not a.phonemes:
        ap.error("give some text or --phonemes")
    import numpy as np
    import soundfile as sf
    import torch
    from .synth import Ito
    torch.set_grad_enabled(False)
    from .weights import WeightsAccessError
    t0 = time.time()
    try:
        tts = Ito.load(a.model, a.device, voice=a.voice)
    except WeightsAccessError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    t1 = time.time()
    if a.phonemes:
        parts = [dict(phonemes=a.phonemes)]
    else:
        sents = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", a.text.strip()) if p.strip()]
        parts = [dict(text=s) for s in sents]
    pause = np.zeros(int(0.15 * tts.sr), np.float32)
    out = []
    for i, p in enumerate(parts):
        y = tts.synthesize(**p, seed=a.seed + i, chunk=a.chunk, streaming=a.stream, duration_scale=1.0 / a.speed,
                           style=a.style)
        out += ([pause] if out else []) + [y]
    y = np.concatenate(out)
    sf.write(a.out, y, tts.sr, subtype="PCM_16")
    dt = time.time() - t1
    print(f"{a.out}: {len(y) / tts.sr:.2f} s of audio, {dt:.2f} s to synthesize on {a.device} "
          f"(RTF {dt / (len(y) / tts.sr):.2f}; model load {t1 - t0:.2f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
