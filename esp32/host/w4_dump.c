// Test helper for the int4 path: runs the engine's int4 GEMM (itofs_qgemm4_ref + itofs_w4_unpack_row) on pseudo-random int8 activations for one
// tensor of a blob and writes the exact int32 accumulators, so tools_dev/w4_check.py can compare them with an independent PyTorch/NumPy int64 matmul.
//   w4_dump <blob> <tensor name> <rows> <out prefix>      -> <prefix>.x (int8 rows x ldx), <prefix>.acc (int32 rows x out), prints "K out in rs ldx"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itofs.h"
int main(int argc, char **argv)
{
    if (argc < 5) return 2;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 1;
    fseek(f, 0, SEEK_END); long bs = ftell(f); fseek(f, 0, SEEK_SET);
    void *blob = aligned_alloc(64, ((size_t)bs + 63) & ~63ul); if (fread(blob, 1, (size_t)bs, f) != (size_t)bs) return 1; fclose(f);
    const void *d; size_t nb; int dt;
    if (itofs_blob_find(blob, (size_t)bs, argv[2], &d, &nb, &dt) || dt != 4) { fprintf(stderr, "tensor %s is not an int4 tensor\n", argv[2]); return 1; }
    // shape from the table: entry layout is private to the engine, so read it here the same way (name[64], dtype, ndim, shape[4], ...)
    const uint8_t *b = blob; uint32_t clen = *(const uint32_t *)(b + 8), nt = *(const uint32_t *)(b + 12 + clen);
    int K = 0, out = 0, in = 0;
    for (uint32_t i = 0; i < nt; i++) { const uint8_t *e = b + 16 + clen + (size_t)i * 104; if (!strcmp((const char *)e, argv[2])) { K = *(const int *)(e + 72); out = *(const int *)(e + 76); in = *(const int *)(e + 80); } }
    const int rows = atoi(argv[3]), ldx = (in + 15) & ~15, rs = itofs_w4_row_bytes(in);
    int8_t *x = calloc((size_t)rows * ldx, 1);
    uint32_t s = 12345u;
    for (int r = 0; r < rows; r++) for (int i = 0; i < in; i++) { s = s * 1664525u + 1013904223u; x[(size_t)r * ldx + i] = (int8_t)((int)((s >> 16) % 255) - 127); }
    int32_t *acc = malloc((size_t)rows * out * 4);
    char fn[512];
    snprintf(fn, sizeof fn, "%s.x", argv[4]); f = fopen(fn, "wb"); fwrite(x, 1, (size_t)rows * ldx, f); fclose(f);
    snprintf(fn, sizeof fn, "%s.acc", argv[4]); f = fopen(fn, "wb");
    for (int k = 0; k < K; k++) { itofs_qgemm4_ref(x, rows, ldx, in, (const uint8_t *)d + (size_t)k * out * rs, rs, out, acc, NULL); fwrite(acc, 4, (size_t)rows * out, f); }
    fclose(f);
    printf("%d %d %d %d %d\n", K, out, in, rs, ldx);
    return 0;
}
