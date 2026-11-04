#!/usr/bin/env python3
"""Golden files for the C unit test (esp32/host: make test_frontend), from the Python reference (make_reference.py / make_stream_reference.py
output). Single utterance: cat TAB text TAB n_ids TAB fnv32(ids).  Stream: cat TAB text TAB n_utt TAB len:fnv32,len:fnv32,...
   make_golden.py single ref.jsonl [more.jsonl ...] > golden_single.tsv      make_golden.py stream ref_stream.jsonl > golden_stream.tsv"""
import json, sys
def fnv(ids):
    h = 2166136261
    for x in ids:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return "%08x" % h
mode = sys.argv[1]
for fn in sys.argv[2:]:
    for l in open(fn, encoding="utf-8"):
        d = json.loads(l)
        if "meta" in d: continue
        if mode == "single":
            if len(d["ids"]) > 1000: continue
            print("%s\t%s\t%d\t%s" % (d["cat"], d["text"].replace("\t", " "), len(d["ids"]), fnv(d["ids"])))
        else:
            if any(len(u) > 400 for u in d["utt"]): continue
            print("%s\t%s\t%d\t%s" % (d["cat"], d["text"].replace("\t", " "), len(d["utt"]), ",".join("%d:%s" % (len(u), fnv(u)) for u in d["utt"])))
