#!/usr/bin/env python3
"""End-to-end timing on a board: text -> (front end -> engine) with `textq` (no playback, no I2S) or `textp` (same, with an emulated playout clock),
one JSON line per utterance:  text idx, utterance k, tokens, fe_ms, first chunk (engine), e2e_ms (text end -> first chunk ready, utterance 0 only),
compute_ms, audio_s, RTF, PCM hash, underruns, planned start delay, gap to the previous utterance's playout, memory.
   serial_e2e.py --port P file.tsv out.jsonl [--cmd textp|textq] [--tier K] [--limit N]"""
import argparse, json, re, sys, time, serial
ap = argparse.ArgumentParser()
ap.add_argument("tsv"); ap.add_argument("out"); ap.add_argument("--port", required=True)
ap.add_argument("--cmd", default="textp"); ap.add_argument("--tier", type=int); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--idle", type=float, default=20.0); ap.add_argument("--fastfirst", type=int)
a = ap.parse_args()
ser = serial.Serial(a.port, 115200, timeout=0.2); ser.dtr = False; ser.rts = False
buf = b""
def readlines(t):
    global buf
    d = ser.read(65536)
    if d: buf += d
    out = []
    while b"\n" in buf:
        l, buf = buf.split(b"\n", 1); out.append(l.decode("utf-8", "replace").rstrip("\r"))
    return out
def cmd(c, wait=1.5, until=None, timeout=30):
    ser.write((c + "\n").encode()); t0 = time.time(); got = []
    while time.time() - t0 < timeout:
        ls = readlines(0); got += ls
        if until and any(until(l) for l in got): break
        if not until and time.time() - t0 > wait: break
    return got
cmd("\n", 0.5)
if a.tier is not None:
    g = cmd("tier %d" % a.tier, until=lambda l: l.startswith("TIER") or l.startswith("ERROR") or "READY" in l, timeout=60); time.sleep(0.5)
    print("\n".join(l[:200] for l in g))
if a.fastfirst is not None: cmd("fastfirst %d" % a.fastfirst, 1)
rows = [l.rstrip("\n").split("\t", 1) for l in open(a.tsv, encoding="utf-8") if "\t" in l]
if a.limit: rows = rows[:a.limit]
fo = open(a.out, "w")
nutt = 0
for ti, (cat, text) in enumerate(rows):
    ser.write(("%s %s\n" % (a.cmd, text)).encode()); t0 = time.time()
    fes, tims, paced, e2e, mems = [], [], [], [], []; fe_done = False; last = time.time(); raw = []
    while True:
        ls = readlines(0)
        if ls: last = time.time()
        for l in ls:
            raw.append(l)
            m = re.match(r"FE utt (\d+): (\d+) tokens in ([\d.]+) ms \(([\d.]+) ms after the text arrived\), ids fnv32 (\w+)", l)
            if m: fes.append(dict(k=int(m[1]), n=int(m[2]), fe_ms=float(m[3]), ids_fnv=m[5]))
            m = re.match(r"TIMING text[qp]: (\d+) tokens, style \d+, ([\d.]+) s audio \| first chunk ([\d.]+) ms .*compute ([\d.]+) ms, RTF ([\d.]+) \| underruns (\d+) \(start delay (\w+) (-?[\d.]+) ms.*weight set (\d+) \| split (\w+).*pcm fnv32 (\w+)", l)
            if m: tims.append(dict(audio_s=float(m[2]), ttfa_ms=float(m[3]), compute_ms=float(m[4]), rtf=float(m[5]), underruns_play=int(m[6]), plan_delay_ms=float(m[8]), set=int(m[9]), split=m[10], pcm_fnv=m[11]))
            m = re.match(r"PACED: utterance (?:\(first\) |ready )?([\d.]+) ms (\S+) .*underruns (\d+), worst lateness ([\d.]+) ms", l)
            if m: paced.append(dict(gap_abs_ms=float(m[1]), gap_kind=m[2], p_underruns=int(m[3]), p_late_ms=float(m[4])))
            m = re.match(r"TEXT_E2E: text arrived -> first audio chunk ready ([\d.]+) ms \(front end \+ queue ([\d.]+) ms, engine ([\d.]+) ms\)", l)
            if m: e2e.append(dict(e2e_ms=float(m[1]), fe_queue_ms=float(m[2])))
            m = re.match(r"MEM text[qp]: peak PSRAM used (\d+) / \d+ KB, peak internal SRAM used (\d+) / \d+ KB \| free now: PSRAM (\d+) KB, internal SRAM (\d+) KB", l)
            if m: mems.append(dict(peak_psram_kb=int(m[1]), peak_int_kb=int(m[2]), free_psram_kb=int(m[3]), free_int_kb=int(m[4])))
            if l.startswith("FE done"): fe_done = True
            if l.startswith("ERROR"): fe_done = True; print("ERR", l)
        if fe_done and len(tims) >= len(fes) and (a.cmd != "textp" or len(paced) >= len(tims)) and (not tims or len(mems) >= len(tims)):
            time.sleep(0.05); break
        if time.time() - last > a.idle: print("TIMEOUT text", ti); break
    wall = time.time() - t0
    for k, f in enumerate(fes):
        r = dict(text_idx=ti, cat=cat, text=text if k == 0 else None, **f)
        if k < len(tims): r.update(tims[k])
        if k < len(paced): r.update(paced[k])
        if k == 0 and e2e: r.update(e2e[0])
        if k < len(mems): r.update(mems[k])
        if k == 0: r["wall_s"] = wall
        fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush(); nutt += 1
    print(ti, cat, "utts", len(fes), "e2e", e2e[0]["e2e_ms"] if e2e else None, "rtf", [t["rtf"] for t in tims][:3], "und", [p["p_underruns"] for p in paced][:3], flush=True)
