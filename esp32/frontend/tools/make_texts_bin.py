#!/usr/bin/env python3
"""Pack texts for the stand-alone test (partition `texts`, flash at 0xF80000):  make_texts_bin.py out.bin file.tsv [file2.tsv ...]
Each input line is  category<TAB>text  (the category is dropped).  Format: "ITXT", u32 count, u32 payload bytes, u32 0, then per text u16 length + UTF-8 bytes."""
import struct, sys
texts = []
for fn in sys.argv[2:]:
    for l in open(fn, encoding="utf-8"):
        l = l.rstrip("\n")
        if "\t" in l: texts.append(l.split("\t", 1)[1])
body = b"".join(struct.pack("<H", len(t.encode())) + t.encode() for t in texts)
assert all(len(t.encode()) < 4000 for t in texts) and len(body) < 0x20000 - 16
open(sys.argv[1], "wb").write(b"ITXT" + struct.pack("<III", len(texts), len(body), 0) + body)
print(len(texts), "texts,", len(body), "bytes ->", sys.argv[1])
