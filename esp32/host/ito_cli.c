// Host build of the on-chip engine: token ids in, 24 kHz WAV out, with the chip's exact arithmetic (the PCM is
// bit-identical to the firmware's for the same ids, style and seed; see README "What was verified").
//
//   make && ../tools/fetch_weights.sh && ./ito_cli ../../models/ito_female_esp32s3.bin "0,50,51,158,..." out.wav [--seed N] [--act 8|16] [--chunk F]
//   python3 ../tools/say.py "Any English text." --dry     # prints the ids for a sentence (no board needed)
//
// Prints the work done before the first audio chunk and per second of audio (executed int8 MACs), which is what the
// chip's time-to-first-audio and real-time factor depend on (esp32/tools/estimate_v3.py turns them into estimates).
#include "itofs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void write_wav(const char *path, const int16_t *x, long n, int sr)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    const uint32_t data = (uint32_t)n * 2, riff = 36 + data, fmt = 16, rate = (uint32_t)sr, br = (uint32_t)sr * 2;
    const uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    fwrite(x, 2, (size_t)n, f);
    fclose(f);
}

static double total(const double *g) { double s = 0; for (int i = 0; i < ITOFS_G_N; i++) s += g[i]; return s; }

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <weights.bin> <ids: \"0,72,156,...\" or @file> <out.wav> [--seed N] [--act 8|16] [--chunk F]\n", argv[0]);
        return 2;
    }
    uint32_t seed = 1; int act = 8, chunk = 0;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--seed")) seed = (uint32_t)strtoul(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--act")) act = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--chunk")) chunk = atoi(argv[i + 1]);
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long bs = ftell(f); fseek(f, 0, SEEK_SET);
    void *blob = NULL;
    if (posix_memalign(&blob, 64, (size_t)bs) || fread(blob, 1, (size_t)bs, f) != (size_t)bs) return 1;
    fclose(f);
    itofs_model_t m;
    int e = itofs_model_init(&m, blob, (size_t)bs);
    if (e) { fprintf(stderr, "model: %s\n", itofs_strerror(e)); return 1; }

    static char buf[1 << 16];
    const char *ids = argv[2];
    if (ids[0] == '@') {
        FILE *g = fopen(ids + 1, "r");
        if (!g) { perror(ids + 1); return 1; }
        size_t k = fread(buf, 1, sizeof buf - 1, g); buf[k] = 0; fclose(g); ids = buf;
    }
    int tok[400], n = 0;
    for (const char *p = ids; *p && n < 400;) {
        char *end; long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        tok[n++] = (int)v; p = end;
    }
    if (n < 2) { fprintf(stderr, "need at least 2 token ids\n"); return 1; }

    if (chunk <= 0) chunk = 2400 / m.hop > 0 ? 2400 / m.hop : 1;   // 100 ms, as the firmware
    itofs_limits_t lim = { 400, chunk, act, 0 };
    size_t hb, bb;
    itofs_arena_bytes(&m, &lim, &hb, &bb);
    void *hot = NULL, *bulk = NULL;
    if (posix_memalign(&hot, 64, hb) || posix_memalign(&bulk, 64, bb)) return 1;
    itofs_ctx_t c;
    if ((e = itofs_init(&c, &m, &lim, hot, hb, bulk, bb))) { fprintf(stderr, "init: %s\n", itofs_strerror(e)); return 1; }

    const clock_t t0 = clock();
    if ((e = itofs_begin(&c, tok, n, 0, seed)) < 0) { fprintf(stderr, "begin: %s\n", itofs_strerror(e)); return 1; }
    const size_t cap = (size_t)1 << 23;
    int16_t *pcm = malloc(cap * sizeof(int16_t));
    long pos = 0; double first = -1;
    for (;;) {
        if ((size_t)pos + (size_t)chunk * m.hop > cap) { fprintf(stderr, "utterance too long\n"); return 1; }
        int got = itofs_next_chunk(&c, pcm + pos, chunk * m.hop);
        if (got < 0) { fprintf(stderr, "synthesis: %s\n", itofs_strerror(got)); return 1; }
        if (got == 0) break;
        if (first < 0) first = total(c.macs_exec);
        pos += got;
    }
    const double host_s = (double)(clock() - t0) / CLOCKS_PER_SEC, audio_s = (double)pos / m.sr;
    write_wav(argv[3], pcm, pos, m.sr);
    const double all = total(c.macs_exec);
    printf("%s: %d tokens, %.2f s of audio, %d-bit activations, seed %u\n", argv[3], n, audio_s, act, seed);
    printf("executed int8 MACs: %.1f M before the first %d ms chunk, %.1f M per second of audio\n",
           first / 1e6, chunk * m.hop * 1000 / m.sr, all / audio_s / 1e6);
    printf("host time %.2f s (RTF %.2f on this computer; the chip's speed is estimated in esp32/README.md)\n",
           host_s, host_s / audio_s);
    return 0;
}
