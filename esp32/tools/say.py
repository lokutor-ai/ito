#!/usr/bin/env python3
"""Make the board say any English text.

    python3 esp32/tools/say.py "Any English text you like." --port /dev/cu.usbserial-0001
    python3 esp32/tools/say.py "Any English text you like." --dry          # just print the token ids

Phonemises on this computer exactly like the training corpus (ito/text.py: espeak-ng en-us with stress and
punctuation, nltk word_tokenize, StyleTTS 2's symbol table, leading 0), sends `say <ids>` over serial, and prints the
board's reply (timing lines). Long text is split into sentences; each is sent when the board says READY.
Needs:  pip install pyserial phonemizer espeakng-loader nltk     (no PyTorch needed)
"""
import argparse, re, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
from ito.text import text_to_ids  # noqa: E402

MAX_TOKENS = 400        # the board's limit (and the training corpus' longest sentence)
GROUP_TOKENS = 200      # consecutive short sentences are sent together up to this many tokens


def sentences(text):
    parts = [p.strip() for p in re.split(r'(?<=[.!?;])\s+', text.strip()) if p.strip()]
    pieces = []
    for p in parts:                       # too long for the board: split at the middle word until it fits
        while len(text_to_ids(p)) > MAX_TOKENS:
            words = p.split()
            cut = max(1, len(words) // 2)
            pieces.append(" ".join(words[:cut])); p = " ".join(words[cut:])
        pieces.append(p)
    out = []
    for p in pieces:                      # merge short sentences (one pass per group keeps the melody natural)
        if out and len(text_to_ids(out[-1] + " " + p)) <= GROUP_TOKENS:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def wait_ready(ser, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print("   " + line)
        if line.startswith("READY") or line.startswith("ERROR"):
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--port", required=False)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--style", type=int, default=None, help="style index for blobs with a style bank (the released blob has one fixed style)")
    ap.add_argument("--dry", action="store_true", help="print the ids, do not send")
    a = ap.parse_args()
    chunks = sentences(a.text)
    if a.dry or not a.port:
        for s in chunks:
            print(s, "->", ",".join(map(str, text_to_ids(s))))
        return
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0.2)
    ser.dtr = False; ser.rts = False          # do not reset the board
    time.sleep(0.2); ser.reset_input_buffer()
    if a.style is not None:
        ser.write(f"style {a.style}\n".encode())
        wait_ready(ser, 5)
    for s in chunks:
        ids = text_to_ids(s)
        print(f">> {s}  ({len(ids)} tokens)")
        ser.write(("say " + ",".join(map(str, ids)) + "\n").encode())
        wait_ready(ser)


if __name__ == "__main__":
    main()
