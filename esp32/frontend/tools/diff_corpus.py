#!/usr/bin/env python3
"""Compare the C front end (build/fe_cli) with the Python reference (make_reference.py output) stage by stage.
   diff_corpus.py ref.jsonl cli_out.jsonl [--show N]   -> per-category exact-id match, first mismatches"""
import json, sys, collections
ref = [json.loads(l) for l in open(sys.argv[1])][1:]
out = [json.loads(l) for l in open(sys.argv[2])]
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 10
assert len(ref) == len(out), (len(ref), len(out))
cats = collections.defaultdict(lambda: [0, 0, 0, 0])
bad = []
for r, o in zip(ref, out):
    c = cats[r["cat"]]
    c[0] += 1
    c[1] += r["ids"] == o["ids"]
    c[2] += r["raw"].strip() == o["raw"].strip()
    c[3] += r["tok"] == o["tok"]
    if r["ids"] != o["ids"]: bad.append((r, o))
tot = [sum(v[i] for v in cats.values()) for i in range(4)]
print("%-14s %5s %9s %9s %9s" % ("category", "n", "ids ==", "raw ==", "tok =="))
for k, v in sorted(cats.items()):
    print("%-14s %5d %8.1f%% %8.1f%% %8.1f%%" % (k, v[0], 100 * v[1] / v[0], 100 * v[2] / v[0], 100 * v[3] / v[0]))
print("%-14s %5d %8.2f%% %8.2f%% %8.2f%%" % ("ALL", tot[0], 100 * tot[1] / tot[0], 100 * tot[2] / tot[0], 100 * tot[3] / tot[0]))
for r, o in bad[:show]:
    print("--", r["cat"], "|", r["text"]); print("  norm py:", r["norm"]); print("  norm c :", o["norm"])
    print("  raw  py:", r["raw"]); print("  raw  c :", o["raw"]); print("  tok  py:", r["tok"]); print("  tok  c :", o["tok"])
