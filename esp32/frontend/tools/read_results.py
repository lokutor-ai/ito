#!/usr/bin/env python3
"""Decode the `results` partition of the stand-alone test (esptool read-flash 0xFA0000 0x40000 results.bin) into JSON lines.
   read_results.py results.bin [out.jsonl]      prints a summary; records:  hdr / boot / utt / done"""
import json, struct, sys
d = open(sys.argv[1], "rb").read()
M = {0x31485249: "hdr", 0x31544249: "boot", 0x31545549: "utt", 0x314e4449: "done"}
recs = []
off = 0
def one(o):
    m = struct.unpack_from("<I", d, o)[0]
    if m == 0xFFFFFFFF: return None
    k = M.get(m)
    if k == "utt":
        (text, utt, boot, ntok, ids, pcm, samples, fe, e2e, ttfa, comp, gap, und, st, fl, fp, fi, mp, mi) = struct.unpack_from("<HHHHIIIIIIIiHBBIIII", d, o + 4)
        return dict(kind="utt", text=text, utt=utt, boot=boot, ntok=ntok, ids_fnv="%08x" % ids, pcm_fnv="%08x" % pcm, samples=samples, fe_us=fe, e2e_us=e2e, ttfa_us=ttfa,
                    compute_us=comp, gap_us=gap, underruns=und, set=st, gap_valid=bool(fl & 1), truncated=bool(fl & 2), full_split=bool(fl & 4),
                    free_psram=fp, free_int=fi, minfree_psram=mp, minfree_int=mi)
    if k in ("hdr", "boot", "done"):
        v = struct.unpack_from("<15I", d, o + 4)
        names = "boot reset_reason flags set first_text n_texts calib_rtf_milli free_psram free_int minfree_psram minfree_int uptime_ms fe_stack_hw texts_fnv".split()
        r = dict(kind=k, **dict(zip(names, v[:14])))
        r["texts_fnv"] = "%08x" % r["texts_fnv"]
        return r
    return dict(kind="garbage", magic="%08x" % m)
o = 0
if len(d) >= 64:
    h = one(0); recs.append(h) if h else None
o = 4096
while o + 64 <= len(d):
    r = one(o)
    if r is None: break
    recs.append(r); o += 64
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
from collections import Counter
print(Counter(r["kind"] for r in recs))
for r in recs:
    if r["kind"] in ("hdr", "boot", "done"): print(r)
