"""Rewrite an ItoFS blob with int4 (dtype 4) tensors as the equivalent plain int8 blob: every int4 weight row is unpacked to its int8 values
w8 = (q - zp) * m (the same arithmetic as itofs_w4_unpack_row) and stored as dtype 1 with the same per-channel scales. The result must give
BIT-IDENTICAL engine output to the int4 blob: that is the int4 path's exactness check.
    python3 i4_to_i8.py <int4.bin> <int8_equivalent.bin>"""
import sys, struct
import numpy as np
G4 = 32

def unpack_int4(data, K, out, nin):
    ng = -(-nin // G4); rb = ng * 16 + ((ng + 15) // 16) * 16
    a = np.frombuffer(data, np.uint8).reshape(K, out, rb)
    res = np.zeros((K, out, ng * G4), np.int64)
    for g in range(ng):
        pb = a[:, :, ng * 16 + g].astype(np.int64); m, zp = pb >> 4, pb & 15
        nb = a[:, :, g * 16:(g + 1) * 16].astype(np.int64)
        res[:, :, g * G4:g * G4 + 16] = ((nb & 15) - zp[..., None]) * m[..., None]
        res[:, :, g * G4 + 16:(g + 1) * G4] = ((nb >> 4) - zp[..., None]) * m[..., None]
    return res[:, :, :nin]

def main():
    b = open(sys.argv[1], "rb").read()
    magic, ver, clen = struct.unpack_from("<4sII", b, 0)
    assert magic == b"ITF1"
    (n,) = struct.unpack_from("<I", b, 12 + clen)
    ents, pos, n4 = [], 16 + clen, 0
    for _ in range(n):
        name, dt, nd, s0, s1, s2, s3, off, nb, soff, flags = struct.unpack_from("<64sII4IIIII", b, pos); pos += 104
        nm = name.rstrip(b"\0").decode(); shape = (s0, s1, s2, s3)[:nd]
        sc = b[soff:soff + 4 * s1] if dt in (1, 3, 4) else None
        data = b[off:off + nb]
        if dt == 4:
            w8 = unpack_int4(data, s0, s1, s2); assert np.abs(w8).max() <= 127
            data, dt = w8.astype(np.int8).tobytes(), 1; n4 += 1
        ents.append((nm, dt, shape, data, sc, flags))
    head = struct.pack("<4sII", b"ITF1", 1, clen) + b[12:12 + clen] + struct.pack("<I", len(ents))
    align = lambda x: (x + 63) & ~63
    table_off = len(head); p = align(table_off + 104 * len(ents)); table, blobs = b"", []
    for nm, dt, shape, data, sc, flags in ents:
        off = p; p = align(p + len(data)); blobs.append((off, data)); soff = 0
        if sc is not None: soff = p; p = align(p + len(sc)); blobs.append((soff, sc))
        shp = list(shape) + [0] * (4 - len(shape))
        table += struct.pack("<64sII4IIIII", nm.encode(), dt, len(shape), *shp, off, len(data), soff, flags)
    out = bytearray(p); out[:len(head)] = head; out[table_off:table_off + len(table)] = table
    for off, data in blobs: out[off:off + len(data)] = data
    open(sys.argv[2], "wb").write(bytes(out))
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {n4} int4 tensors unpacked, {len(out)} bytes")
main()
