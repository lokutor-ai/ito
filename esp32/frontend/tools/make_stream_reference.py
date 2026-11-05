#!/usr/bin/env python3
"""Reference for the stream splitter: what tools/say.py would send for a text (its sentences() function + text_to_ids).
   make_stream_reference.py multi.tsv out.jsonl [--fastfirst]
--fastfirst: the chip's default (ITOFS_FE_FAST_FIRST): the first sentence is its own utterance, the rest is grouped exactly as say.py does."""
import json, sys, os
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "../../.."))
sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "esp32/tools"))
import say
from ito.text import text_to_ids
import nltk
from nltk.tokenize import word_tokenize
word_tokenize("a b. c")
FAST = "--fastfirst" in sys.argv
def pieces_of(text):
    import re
    parts = [p.strip() for p in re.split(r'(?<=[.!?;])\s+', text.strip()) if p.strip()]
    pieces = []
    for p in parts:
        while len(text_to_ids(p)) > say.MAX_TOKENS:
            words = p.split(); cut = max(1, len(words) // 2)
            pieces.append(" ".join(words[:cut])); p = " ".join(words[cut:])
        pieces.append(p)
    return pieces
def group(pieces):
    out = []
    for p in pieces:
        if out and len(text_to_ids(out[-1] + " " + p)) <= say.GROUP_TOKENS: out[-1] = out[-1] + " " + p
        else: out.append(p)
    return out
def utterances(text):
    if not FAST: return say.sentences(text)
    pcs = pieces_of(text)
    return pcs[:1] + group(pcs[1:])
with open(sys.argv[2], "w") as fo:
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.rstrip("\n")
        if not line: continue
        cat, _, text = line.partition("\t")
        utts = [text_to_ids(u) for u in utterances(text)]
        fo.write(json.dumps({"cat": cat, "text": text, "utt": utts}, ensure_ascii=False) + "\n")
