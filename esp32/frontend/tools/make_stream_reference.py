#!/usr/bin/env python3
"""Reference for the stream splitter: what tools/say.py would send for a text (its sentences() function + text_to_ids).
   make_stream_reference.py multi.tsv out.jsonl"""
import json, sys, os
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "../../.."))
sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "esp32/tools"))
import say
from ito.text import text_to_ids
import nltk
from nltk.tokenize import word_tokenize
word_tokenize("a b. c")
with open(sys.argv[2], "w") as fo:
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.rstrip("\n")
        if not line: continue
        cat, _, text = line.partition("\t")
        utts = [text_to_ids(u) for u in say.sentences(text)]
        fo.write(json.dumps({"cat": cat, "text": text, "utt": utts}, ensure_ascii=False) + "\n")
