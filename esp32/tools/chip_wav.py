#!/usr/bin/env python3
"""Any English text -> WAV with the chip's exact arithmetic, on this computer (no board needed).

    cd esp32/host && make ito_cli && cd ../..
    python3 esp32/tools/chip_wav.py "Good morning! The coffee is ready." out.wav
    python3 esp32/tools/chip_wav.py --voice g "Good morning! The coffee is ready." out_g.wav   # male voice

Phonemises like tools/say.py, then runs esp32/host/ito_cli (the firmware's C engine built for the host; its PCM is
bit-identical to the firmware's for the same ids and seed). Sentences are synthesized one by one and joined with a
150 ms pause."""
import argparse, os, re, subprocess, sys, tempfile, wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ito.text import text_to_ids  # noqa: E402
from ito.weights import weights_path, voice_files, WeightsAccessError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--voice", default="d", type=str.lower, choices=("d", "g"), help="d: female (default); g: male")
    ap.add_argument("--blob", default=None, help="chip weights (default: $ITO_WEIGHTS_DIR, models/, or Hugging Face)")
    ap.add_argument("--cli", default=os.path.join(ROOT, "esp32", "host", "ito_cli"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--act", type=int, default=8, choices=(8, 16))
    a = ap.parse_args()
    if a.blob is None:
        try:
            a.blob = weights_path(voice_files(a.voice)[1])
        except WeightsAccessError as e:
            sys.exit(str(e))
    if not os.path.exists(a.cli):
        sys.exit(f"{a.cli} not found: run `make` in esp32/host first")
    sents = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", a.text.strip()) if p.strip()]
    frames, sr = [], 24000
    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(sents):
            ids = text_to_ids(s)
            if len(ids) > 400:
                sys.exit(f"sentence too long for the chip ({len(ids)} > 400 tokens): {s[:60]}...")
            w = os.path.join(td, f"{i}.wav")
            r = subprocess.run([a.cli, a.blob, ",".join(map(str, ids)), w, "--seed", str(a.seed + i), "--act", str(a.act)],
                               capture_output=True, text=True, check=True)
            print(r.stdout.strip().splitlines()[1] if len(r.stdout.strip().splitlines()) > 1 else r.stdout.strip())
            with wave.open(w) as f:
                sr = f.getframerate()
                if frames:
                    frames.append(b"\0\0" * int(0.15 * sr))
                frames.append(f.readframes(f.getnframes()))
    with wave.open(a.out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(b"".join(frames))
    print(f"{a.out}: {sum(len(x) for x in frames) / 2 / sr:.2f} s")


if __name__ == "__main__":
    main()
