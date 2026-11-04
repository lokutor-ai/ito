#!/usr/bin/env python3
"""Pack the espeak-ng 1.52.0 data files needed for en-us text -> phonemes into one blob (see espeak/src/ito_vfs.h).
   pack_data.py <espeak-ng-data dir> <out.bin>"""
import struct, sys, os
FILES = ["phontab", "phonindex", "phondata", "intonations", "en_dict", "lang/gmw/en-US"]
# phondata holds the waveform data of the synthesizer; the phoneme programs live in phonindex. The translator only reads the
# 8-byte header (version, sample rate) of phondata: with the rest cut, the output is identical (checked on the corpus and on
# 18000 random lines), and the blob is 273 KB instead of 828 KB. --full keeps all of it.
full = "--full" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
src, out = args[0], args[1]
datas = [open(os.path.join(src, f), "rb").read() for f in FILES]
if not full:
    datas[FILES.index("phondata")] = datas[FILES.index("phondata")][:8]
hdr = 12 + 36 * len(FILES)
off = (hdr + 3) & ~3
ents = b""
body = b""
for f, d in zip(FILES, datas):
    ents += struct.pack("<28sII", f.encode(), off + len(body), len(d))
    body += d + b"\0" * ((-len(d)) % 4)
blob = b"ITFE" + struct.pack("<II", 1, len(FILES)) + ents
blob += b"\0" * (off - len(blob)) + body
open(out, "wb").write(blob)
print(out, len(blob), "bytes")
