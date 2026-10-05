# ItoFS weight blob (`ITF1`)

One file per weight set. The firmware and `engine/itofs.c` read it in place (no parsing copy); every tensor starts on a 64-byte boundary.

```
offset 0   char[4]   "ITF1"
       4   u32       format version: 1, or 2 when the blob holds int4 tensors (an engine that only knows version 1 refuses a version-2 blob)
       8   u32       cfg_len
      12   char[]    cfg: "key=value\n" lines (the model's dimensions: arch, n_vocab, dec_dim, dec_inter, dec_blocks, ...)
12+cfg_len u32       n_tensors
16+cfg_len n_tensors x 104-byte table entries:
                     char[64] name | u32 dtype | u32 ndim | u32 shape[4] | u32 data offset | u32 data bytes | u32 scale offset | u32 flags
```

| dtype | meaning |
|---|---|
| 0 | float32, torch layout |
| 1 | int8 weights, shape `[K][out][in]` (tap-major), per-output-channel float32 scales at the scale offset; flag bit 0: the layer takes 15-bit activations |
| 2 | int32 records: `selftest`, `selftest_a16` (golden-sentence tokens and the fnv32 hash of the host engine's PCM), `demos`, `demo_text` |
| 3 | int16 weights as two int8 planes (high plane, then low plane), per-channel scales |
| **4** | **int4 weights (format version 2)**, shape `[K][out][in]`, per-channel float32 scales (the "base" scale) |

## int4 rows (dtype 4)

Each weight row of `in` values is stored in groups of 32 along the input axis. With `ng = ceil(in / 32)`:

```
row = [ng * 16 bytes of nibbles][ng bytes of group parameters, padded to a multiple of 16]      row bytes = ng * 16 + 16 * ceil(ng / 16)
```

Group `g` has 16 bytes of nibbles: byte `j` holds weight `j` of the group in its low nibble and weight `16 + j` in its high nibble (`q` = 0..15).
Its parameter byte is `(m << 4) | zp`, with the integer multiplier `m` = 1..7 and the zero point `zp` = 0..15. The int8 weight is

```
w8 = (q - zp) * m              |w8| <= 105
```

and the layer's float scale applies exactly as for int8 weights. So an int4 layer is an int8 layer whose weights have this form: the int32 accumulators, the
rescale and the audio are those of the equivalent int8 blob, bit for bit (`esp32/tools/int4_check.py` checks this). Weights past `in` in the last group are 0. Row starts are
16-byte aligned, which the firmware relies on for 32-bit loads. Storage is 0.53 bytes per weight at 32 per group (0.5 + 1/32), plus up to 15 bytes of padding per row.

The unpack (`itofs_w4_unpack_row`, and the firmware's copy of it on 32-bit words): for the 4 weights in one byte lane of a word,
`((word & 0x0F0F0F0F) * m + (128 - zp * m) * 0x01010101) ^ 0x80808080` (low nibbles) and the same on `(word >> 4) & 0x0F0F0F0F` (high nibbles): every
byte is `q * m + 128 - zp * m` in 23..233, so there is no carry between bytes, and the XOR turns the biased byte into the two's-complement `w8`.
