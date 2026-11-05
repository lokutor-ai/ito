// Batch host engine for the on-silicon comparison: loads a weight blob once, then for every stdin line "SEED id,id,id,..." synthesizes
// the utterance with the firmware's arithmetic and prints "FNV32(hex) NSAMPLES AUDIO_SECONDS". The hash is the firmware's PCM hash
// (FNV-1a over the int16 samples, as 16-bit words, see speak() in firmware/main/main.c).
//   ./ito_batch ../../models/ito_female_esp32s3.bin < lines > hashes
#include "itofs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin < lines\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long bs = ftell(f); fseek(f, 0, SEEK_SET);
    void *blob = NULL;
    if (posix_memalign(&blob, 64, (size_t)bs) || fread(blob, 1, (size_t)bs, f) != (size_t)bs) return 1;
    fclose(f);
    itofs_model_t m;
    int e = itofs_model_init(&m, blob, (size_t)bs);
    if (e) { fprintf(stderr, "model: %s\n", itofs_strerror(e)); return 1; }
    const int chunk = 2400 / m.hop > 0 ? 2400 / m.hop : 1;
    itofs_limits_t lim = { 400, chunk, 8, 0 };
    size_t hb, bb;
    itofs_arena_bytes(&m, &lim, &hb, &bb);
    void *hot = NULL, *bulk = NULL;
    if (posix_memalign(&hot, 64, hb) || posix_memalign(&bulk, 64, bb)) return 1;
    itofs_ctx_t c;
    if ((e = itofs_init(&c, &m, &lim, hot, hb, bulk, bb))) { fprintf(stderr, "init: %s\n", itofs_strerror(e)); return 1; }
    const size_t cap = (size_t)1 << 21;
    int16_t *pcm = malloc(cap * sizeof(int16_t));
    static char line[1 << 16];
    while (fgets(line, sizeof line, stdin)) {
        char *p; unsigned long seed = strtoul(line, &p, 10);
        int tok[400], n = 0;
        while (*p && n < 400) { char *end; long v = strtol(p, &end, 10); if (end == p) { p++; continue; } tok[n++] = (int)v; p = end; }
        if (n < 2) { printf("ERR tokens\n"); fflush(stdout); continue; }
        int T = itofs_begin(&c, tok, n, 0, (uint32_t)seed);
        if (T < 0) { printf("ERR %s\n", itofs_strerror(T)); fflush(stdout); continue; }
        long pos = 0; uint32_t h = 2166136261u;
        for (;;) {
            if ((size_t)pos + (size_t)chunk * m.hop > cap) break;
            int got = itofs_next_chunk(&c, pcm + pos, chunk * m.hop);
            if (got <= 0) break;
            for (int i = 0; i < got; i++) { h ^= (uint16_t)pcm[pos + i]; h *= 16777619u; }
            pos += got;
        }
        printf("%08x %ld %.3f\n", h, pos, (double)pos / m.sr); fflush(stdout);
    }
    return 0;
}
