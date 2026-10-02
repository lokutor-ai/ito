// Self-test reference for the firmware: synthesises golden sentence 0 with the HOST C engine exactly as the firmware
// does at boot (seeded source noise, chunk = 2400 / hop frames, int16 output) and writes
//   <prefix>.pcm   int16 PCM (little endian), for the sample-by-sample QEMU comparison
//   <prefix>.i32   int32 record: 'STR2', n_tokens, style_idx, seed, n_samples, fnv32(pcm), act_bits, tokens...
// The exporter stores the record in the weight blob as tensor "selftest" (dtype 2 = int32), so one app binary
// verifies whichever blob is flashed.
//   ./gen_selftest <itofs_weights.bin> <golden/ref0_w8a8.bin> <prefix> [seed] [act_bits 8|16]
#include "itofs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *rd(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void *b = NULL;
    if (posix_memalign(&b, 64, *n + 64)) return NULL;
    if (fread(b, 1, *n, f) != *n) return NULL;
    fclose(f);
    return b;
}

// records: u32 name_len | name | u32 dtype | u32 count | data
static const void *rec(const unsigned char *b, size_t sz, const char *name, int *cnt)
{
    size_t pos = 4;
    while (pos + 12 <= sz) {
        uint32_t nl, dt, n;
        memcpy(&nl, b + pos, 4);
        const char *nm = (const char *)b + pos + 4;
        memcpy(&dt, b + pos + 4 + nl, 4); memcpy(&n, b + pos + 8 + nl, 4);
        const void *d = b + pos + 12 + nl;
        if (strlen(name) == nl && !memcmp(nm, name, nl)) { *cnt = (int)n; return d; }
        pos += 12 + nl + (size_t)n * (dt == 2 ? 8 : 4);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <blob> <ref0_w8a16.bin> <prefix> [seed]\n", argv[0]); return 2; }
    size_t bs, rs;
    void *blob = rd(argv[1], &bs);
    unsigned char *r = rd(argv[2], &rs);
    if (!blob || !r) return 1;
    itofs_model_t m;
    if (itofs_model_init(&m, blob, bs)) return 1;
    int n, ns;
    const int *tok = rec(r, rs, "tokens", &n), *sidx = rec(r, rs, "style_idx", &ns);
    if (!tok || !sidx) return 1;
    const uint32_t seed = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1u;
    const int act = argc > 5 ? atoi(argv[5]) : 8;
    const int chunk = 2400 / m.hop > 0 ? 2400 / m.hop : 1;
    itofs_limits_t lim = { 400, chunk, act, 0 };
    size_t hb, bb;
    itofs_arena_bytes(&m, &lim, &hb, &bb);
    itofs_ctx_t c;
    void *h = malloc(hb), *b = malloc(bb);
    if (itofs_init(&c, &m, &lim, h, hb, b, bb)) return 1;
    int T = itofs_begin(&c, tok, n, sidx[0], seed);
    if (T < 0) return 1;
    const size_t cap = (size_t)1 << 23;      // T is not known up front with the incremental text side
    int16_t *pcm = malloc(sizeof(int16_t) * cap);
    long pos = 0;
    for (;;) { if ((size_t)pos + (size_t)chunk * m.hop > cap) return 1; int got = itofs_next_chunk(&c, pcm + pos, chunk * m.hop); if (got <= 0) break; pos += got; }
    uint32_t hsh = 2166136261u;
    for (long i = 0; i < pos; i++) { hsh ^= (uint16_t)pcm[i]; hsh *= 16777619u; }
    char path[1024];
    snprintf(path, sizeof path, "%s.pcm", argv[3]);
    FILE *f = fopen(path, "wb"); fwrite(pcm, 2, (size_t)pos, f); fclose(f);
    snprintf(path, sizeof path, "%s.i32", argv[3]);
    f = fopen(path, "wb");
    int32_t hdr[7] = { 0x32525453, n, sidx[0], (int32_t)seed, (int32_t)pos, (int32_t)hsh, act };
    fwrite(hdr, 4, 7, f); fwrite(tok, 4, (size_t)n, f); fclose(f);
    printf("selftest ref: %d tokens, style %d, seed %u, act %d bit, chunk %d frames, %ld samples, fnv32 %08x\n", n, sidx[0], seed, act, chunk, pos, hsh);
    return 0;
}
