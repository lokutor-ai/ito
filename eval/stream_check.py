"""Streaming check on the released model, on the CPU, no corpus needed: for a few sentences, synthesize the whole
utterance at once and through the chunked streaming path, and report the SNR between them and the lookahead used.
Negative controls remove one frame of lookahead from one stage; the output must then differ (low SNR), which shows
the lookahead is exactly what each stage needs.

    python eval/stream_check.py [--chunk 8]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ito import Ito  # noqa: E402

SENTENCES = ["Good morning! The coffee is ready.",
             "Could you grab some quinoa and Worcestershire sauce on your way home?",
             "When I finally got to the station, the last train had already left."]


def snr(a, b):
    e = (a - b).pow(2).sum().item()
    return 999.0 if e == 0 else 10 * np.log10(a.pow(2).sum().item() / e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tts = Ito.load(a.model)
    ch = tts.chain
    worst = 999.0
    for k, text in enumerate(SENTENCES):
        hf, s, p0, nz = tts._prepare(tts.tokens(text), seed=k, duration_scale=1.0)
        yf, _, _ = ch.infer_full(hf, s, p0, nz)
        ys, la = ch.infer_stream(hf, s, p0, nz, chunk=a.chunk)
        r = snr(yf, ys); worst = min(worst, r)
        line = f"{text[:48]:48s} T={hf.size(-1):4d} frames  stream vs whole {r:6.1f} dB  lookahead {la} frames"
        if k == 0:
            neg = {n: snr(yf, ch.infer_stream(hf, s, p0, nz, chunk=a.chunk, **{n: 1})[0])
                   for n in ("cut_istft", "cut_voc", "cut_mel", "cut_pros")}
            line += "\n  negative controls (one frame less lookahead): " + ", ".join(f"{n[4:]} {v:.1f} dB" for n, v in neg.items())
        print(line, flush=True)
    print(f"worst stream-vs-whole SNR {worst:.1f} dB ({'PASS' if worst > 90 else 'FAIL'}: float32 rounding only "
          f"when > 90 dB); lookahead {la} frames = {la * 1000 // 80} ms of features")


if __name__ == "__main__":
    main()
