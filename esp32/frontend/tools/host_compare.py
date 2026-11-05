#!/usr/bin/env python3
"""Compare per-utterance results from a board (ids hash, PCM hash) with the host: Python front end (stream reference) + the C engine (esp32/host/ito_batch).
   host_compare.py --ref ref_stream.jsonl --chip chip.jsonl --weights0 w0.bin [--weights1 w1.bin] [--weights2 w2.bin] [--out report.json]
chip.jsonl: lines with text_idx (or text) , k (or utt), ids_fnv, pcm_fnv, set  (serial_e2e.py output or read_results.py output; latest line per (text, utt) wins).
The reference line i is the text with index i; the board numbers its utterances as the front end splits them (fast-first)."""
import argparse, json, subprocess, sys, os, collections
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--chip", required=True); ap.add_argument("--out")
for i in range(3): ap.add_argument("--weights%d" % i)
ap.add_argument("--batch", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../host/ito_batch"))
a = ap.parse_args()
ref = [json.loads(l) for l in open(a.ref) if l.strip() and "meta" not in l[:12]]
rows = {}
for l in open(a.chip):
    r = json.loads(l)
    if r.get("kind", "utt") != "utt": continue
    ti = r["text_idx"] if "text_idx" in r else r["text"]
    k = r["k"] if "k" in r else r["utt"]
    rows[(ti, k)] = r
def fnv(ids):
    h = 2166136261
    for x in ids: h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return "%08x" % h
ids_ok = ids_bad = pcm_ok = pcm_bad = missing = 0
jobs = collections.defaultdict(list)       # set -> [(key, line)]
bad_ids = []
for (ti, k), r in sorted(rows.items()):
    utts = ref[ti]["utt"]
    if k >= len(utts): ids_bad += 1; bad_ids.append((ti, k, "extra utterance")); continue
    ids = utts[k]
    if len(ids) > 400: continue
    if fnv(ids) == r["ids_fnv"]: ids_ok += 1
    else: ids_bad += 1; bad_ids.append((ti, k, "ids hash differs"))
    jobs[r["set"]].append(((ti, k), "%d %s" % (1 + k, ",".join(map(str, ids)))))
nref_utts = sum(len([u for u in r["utt"] if len(u) <= 400]) for r in ref[:max(t for t, _ in rows) + 1]) if rows else 0
for st, js in sorted(jobs.items()):
    w = getattr(a, "weights%d" % st)
    if not w: print("no weights for set", st); continue
    p = subprocess.run([a.batch, w], input="\n".join(l for _, l in js) + "\n", capture_output=True, text=True)
    out = p.stdout.strip().split("\n")
    assert len(out) == len(js), (len(out), len(js), p.stderr[:200])
    for (key, _), o in zip(js, out):
        h = o.split()[0]
        if h == rows[key]["pcm_fnv"]: pcm_ok += 1
        else: pcm_bad += 1; print("PCM MISMATCH", key, h, rows[key]["pcm_fnv"])
print("utterances compared: %d | ids identical %d, different %d | PCM identical %d, different %d" % (len(rows), ids_ok, ids_bad, pcm_ok, pcm_bad))
for b in bad_ids[:10]: print("IDS", b)
if a.out: json.dump(dict(utterances=len(rows), ids_ok=ids_ok, ids_bad=ids_bad, pcm_ok=pcm_ok, pcm_bad=pcm_bad), open(a.out, "w"))
