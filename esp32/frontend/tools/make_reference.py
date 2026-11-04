#!/usr/bin/env python3
"""Reference front end: runs the repository's Python front end (ito/text.py: normalize -> phonemizer/espeak-ng en-us ->
nltk word_tokenize -> StyleTTS 2 symbol ids) over a corpus and writes one JSON object per line:
    {"cat", "text", "norm", "raw" (phonemizer output), "tok" (after word_tokenize), "ids"}
The nltk tokenizer is REQUIRED here: ito/text.py silently falls back to a regex approximation if nltk fails to import,
which would make the reference wrong, so this script refuses to run without it.
   make_reference.py corpus.tsv out.jsonl [--repo /path/to/ito-release]"""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--espeak-parent", default=None, help="old phonemizer (<3.4): directory that CONTAINS espeak-ng-data, kept in ESPEAK_DATA_PATH")
ap.add_argument("--legacy-punct", action="store_true", help="emulate phonemizer <= 3.3.0 punctuation handling: plain mark regex, no inline-group (quote/bracket) context")
ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
a = ap.parse_args()
sys.path.insert(0, a.repo)
if a.espeak_parent:                     # ito.text points ESPEAK_DATA_PATH at the data dir itself (right for phonemizer >= 3.4)
    os.environ["ESPEAK_DATA_PATH"] = a.espeak_parent
    _set = os.environ.__class__.__setitem__
    os.environ.__class__.__setitem__ = lambda self, k, v: None if k == "ESPEAK_DATA_PATH" else _set(self, k, v)
import nltk
from nltk.tokenize import word_tokenize
try:
    word_tokenize("a b. c")
except LookupError:
    nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)
    word_tokenize("a b. c")
import ito.text as T
T.phonemize("warm up")
if a.legacy_punct:
    import re
    from phonemizer.punctuation import _DEFAULT_MARKS
    T._backend._punctuator._marks_re = re.compile(r"(\s*[%s]+\s*)+" % re.escape(_DEFAULT_MARKS))
    T._backend._punctuation_context = False
import phonemizer
meta = {"phonemizer": phonemizer.__version__, "nltk": nltk.__version__}
with open(a.out, "w") as fo:
    fo.write(json.dumps({"meta": meta}) + "\n")
    for line in open(a.corpus, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line: continue
        cat, _, text = line.partition("\t")
        norm = T.normalize(text).strip().replace('"', "")
        raw = T._backend.phonemize([norm])[0]
        tok = " ".join(word_tokenize(raw))
        ids = T.phonemes_to_ids(tok)
        assert tok == T.phonemize(text)
        fo.write(json.dumps({"cat": cat, "text": text, "norm": norm, "raw": raw, "tok": tok, "ids": ids}, ensure_ascii=False) + "\n")
print("wrote", a.out, meta)
