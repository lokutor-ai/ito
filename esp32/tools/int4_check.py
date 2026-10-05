#!/usr/bin/env python3
"""Exactness checks of a blob with int4 weights (format version 2), no training code and no phonemiser needed:

  1. GEMM: for every int4 tensor, the engine's int32 accumulators (host/w4_dump, i.e. itofs_qgemm4_ref + itofs_w4_unpack_row) equal an independent
     NumPy int64 matmul of random int8 activations with the weights unpacked here (w8 = (q - zp) * m per group of 32 along the input axis).
  2. Audio: the blob is rewritten as the equivalent plain int8 blob (tools/i4_to_i8.py: every int4 row unpacked to its int8 values, same per-channel
     scales); host/ito_cli must give BIT-IDENTICAL PCM for both, at 8- and 16-bit activations, for the blob's self-test and demo sentences.

    cd esp32/host && make && cd ../..
    python3 esp32/tools/int4_check.py models/ito_female_esp32s3_int4.bin
"""
import os, struct, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(HERE, "..", "host")
G4 = 32


def entries(b):
    _, ver, clen = struct.unpack_from("<4sII", b, 0)
    (n,) = struct.unpack_from("<I", b, 12 + clen)
    for i in range(n):
        name, dt, nd, s0, s1, s2, s3, off, nb, soff, fl = struct.unpack_from("<64sII4IIIII", b, 16 + clen + 104 * i)
        yield name.rstrip(b"\0").decode(), dt, (s0, s1, s2, s3)[:nd], off, nb


def check_gemm(path, b):
    bad = tot = 0
    for nm, dt, shape, off, nb in entries(b):
        if dt != 4:
            continue
        K, out, nin = shape
        ng = -(-nin // G4); rb = ng * 16 + ((ng + 15) // 16) * 16
        a = np.frombuffer(b[off:off + nb], np.uint8).reshape(K, out, rb)
        w = np.zeros((K, out, ng * G4), np.int64)
        for g in range(ng):
            pb = a[:, :, ng * 16 + g].astype(np.int64); m, zp = pb >> 4, pb & 15
            q = a[:, :, g * 16:(g + 1) * 16].astype(np.int64)
            w[:, :, g * G4:g * G4 + 16] = ((q & 15) - zp[..., None]) * m[..., None]
            w[:, :, g * G4 + 16:(g + 1) * G4] = ((q >> 4) - zp[..., None]) * m[..., None]
        w = w[:, :, :nin]
        rows = 7
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([os.path.join(HOST, "w4_dump"), path, nm, str(rows), os.path.join(td, "t")], capture_output=True, text=True).stdout.split()
            ldx = int(r[4])
            x = np.fromfile(os.path.join(td, "t.x"), np.int8).reshape(rows, ldx)[:, :nin].astype(np.int64)
            acc = np.fromfile(os.path.join(td, "t.acc"), np.int32).reshape(K, rows, out).astype(np.int64)
        ok = np.array_equal(np.stack([x @ w[k].T for k in range(K)]), acc)
        tot += 1; bad += not ok
        print(f"  {nm:34s} K{K} out{out} in{nin}: C int4 GEMM {'== int64 reference' if ok else 'DIFFERENT'}")
    print(f"GEMM check: {'PASS' if not bad else 'FAIL'} ({tot} int4 tensors)")
    return bad == 0


def sentences(b):
    out = {}
    for nm, dt, shape, off, nb in entries(b):
        if dt == 2 and nm in ("selftest", "demos"):
            out[nm] = np.frombuffer(b[off:off + nb], "<i4")
    s = []
    if "selftest" in out:
        r = out["selftest"]; s.append(list(r[7:7 + r[1]]))
    if "demos" in out:
        r = out["demos"]; p = 1
        for _ in range(int(r[0])):
            n = int(r[p]); s.append(list(r[p + 2:p + 2 + n])); p += 2 + n
    return s


def check_audio(path):
    with tempfile.TemporaryDirectory() as td:
        eq = os.path.join(td, "eq8.bin")
        print(subprocess.run([sys.executable, os.path.join(HERE, "i4_to_i8.py"), path, eq], capture_output=True, text=True).stdout.strip())
        sents = sentences(open(path, "rb").read())
        same_all = True
        for k, ids in enumerate(sents):
            for act in (8, 16):
                pcm = []
                for blob in (path, eq):
                    wav = os.path.join(td, "o.wav")
                    subprocess.run([os.path.join(HOST, "ito_cli"), blob, ",".join(map(str, ids)), wav, "--seed", str(1 + k), "--act", str(act)], check=True, capture_output=True)
                    pcm.append(open(wav, "rb").read())
                same = pcm[0] == pcm[1]; same_all &= same
                if not same:
                    print(f"  sentence {k} ({len(ids)} tokens), {act}-bit activations: int4 and int8-equivalent blobs DIFFER")
        print(f"audio check: {len(sents)} sentences x 8/16-bit activations: int4 blob vs int8-equivalent blob {'BIT-IDENTICAL' if same_all else 'DIFFERENT'}")
        return same_all


if __name__ == "__main__":
    p = sys.argv[1]
    b = open(p, "rb").read()
    ok = check_gemm(p, b)
    ok &= check_audio(p)
    print("int4 check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
