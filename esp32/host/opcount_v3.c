// Host profile of the ItoFS engine's work: int8 GEMM MACs (executed) + the non-GEMM float work (counted with
// ITOFS_OPCOUNT), per second of audio and before the first 100 ms chunk, for the golden sentences, plus an ESP32-S3
// cycle estimate of the non-GEMM part from per-operation cycle costs (ASSUMPTIONS, printed; the board benchmark
// measures the real split: "GEMM ... ms, non-GEMM ... ms").
//   cc -std=c99 -O2 -ffp-contract=off -DITOFS_OPCOUNT -o opcount_v3 engine/itofs.c engine/opcount_v3.c -lm
//   ./opcount_v3 <itofs_weights.bin> <golden dir>
#include "itofs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *opn[ITOFS_OPC_N] = {"expf", "logf", "sincosf", "erf (GELU)", "  of which |x|<=1 series", "activation quantise (elements)",
                                        "LayerNorm (elements)", "FFT (600-point complex)", "Gaussian noise samples", "source samples"};
// LX7 @ 240 MHz cycle-cost assumptions per operation (single-precision FPU, no fast divide; libm-free code)
static const double cyc[ITOFS_OPC_N] = {40, 50, 45, 90, -55, 8, 8, 37000, 120, 70};   // erf: 90 tail, 35 Horner series

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

static double g_wbytes;      // weight bytes read by the GEMM (each call reads its weight block once)
static void qgemm_count(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user)
{
    g_wbytes += (double)in * out;
    itofs_qgemm_ref(x, rows, ldx, in, w, out, acc, user);
}

static void qgemm4_count(const int8_t *x, int rows, int ldx, int in, const uint8_t *w, int rs, int out, int32_t *acc, void *user)
{
    g_wbytes += (double)rs * out;
    itofs_qgemm4_ref(x, rows, ldx, in, w, rs, out, acc, user);
}

static double sum8(const itofs_ctx_t *c) { double a = 0; for (int g = 0; g < ITOFS_G_N; g++) a += c->macs_exec[g]; return a; }
static double sumf(const itofs_ctx_t *c) { double a = 0; for (int g = 0; g < ITOFS_G_N; g++) a += c->macs_f32[g]; return a; }

static double est_cycles(const long long *o, double f32)
{
    double cy = 2.0 * f32;                                // f32 MACs outside the GEMM (depthwise, GRU recurrence, ...)
    for (int k = 0; k < ITOFS_OPC_N; k++) cy += cyc[k] * (double)o[k];
    return cy;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <blob> <golden dir>\n", argv[0]); return 2; }
    size_t bs; void *blob = rd(argv[1], &bs);
    itofs_model_t m;
    if (!blob || itofs_model_init(&m, blob, bs)) { fprintf(stderr, "bad blob\n"); return 1; }
    const int chunk = 2400 / m.hop;
    itofs_limits_t lim = { 400, chunk, 8, 0 };
    size_t hb, bb; itofs_arena_bytes(&m, &lim, &hb, &bb);
    itofs_ctx_t c; void *h = malloc(hb), *b = malloc(bb);
    if (itofs_init(&c, &m, &lim, h, hb, b, bb)) return 1;
    c.qgemm = qgemm_count; c.qgemm4 = qgemm4_count;
    double wb_tot = 0;
    float *pcm = malloc(sizeof(float) * (size_t)chunk * m.hop);
    long long tot[ITOFS_OPC_N] = {0};
    double sec = 0, t8 = 0, tf = 0;
    printf("per sentence: executed int8 GEMM MACs and non-GEMM work (seeded noise, chunk %d frames, 8-bit activations)\n", chunk);
    for (int k = 0; k < 16; k++) {
        char p[1024]; snprintf(p, sizeof p, "%s/ref%d_w8a8.bin", argv[2], k);
        size_t rs; unsigned char *r = rd(p, &rs);
        if (!r) break;
        int n, ns; const int *tok = rec(r, rs, "tokens", &n), *sidx = rec(r, rs, "style_idx", &ns);
        memset(itofs_opc, 0, sizeof itofs_opc);
        memset(c.macs_exec, 0, sizeof c.macs_exec); memset(c.macs_f32, 0, sizeof c.macs_f32);
        g_wbytes = 0;
        itofs_begin(&c, tok, n, sidx[0], 1);
        int got = itofs_next_chunk_f32(&c, pcm, chunk * m.hop);
        const double wb_first = g_wbytes;
        long long first[ITOFS_OPC_N]; memcpy(first, itofs_opc, sizeof first);
        const double f8 = sum8(&c), ff = sumf(&c);
        long samples = got;
        while ((got = itofs_next_chunk_f32(&c, pcm, chunk * m.hop)) > 0) samples += got;
        const double s = (double)samples / m.sr;
        printf("  golden %d: %d tokens, %.2f s | before the first chunk: %.1f M int8 MACs, %.2f M f32 MACs, non-GEMM ~%.1f M cycles, "
               "%.2f MB of weights read | whole: %.1f M int8, non-GEMM ~%.1f M cycles, %.1f MB weights (%.1f MB/s of audio)\n", k, n, s,
               f8 / 1e6, ff / 1e6, est_cycles(first, ff) / 1e6, wb_first / 1e6, sum8(&c) / 1e6, est_cycles(itofs_opc, sumf(&c)) / 1e6,
               g_wbytes / 1e6, g_wbytes / 1e6 / s);
        wb_tot += g_wbytes;
        for (int q = 0; q < ITOFS_OPC_N; q++) tot[q] += itofs_opc[q];
        sec += s; t8 += sum8(&c); tf += sumf(&c);
        free(r);
    }
    printf("\nper second of audio (%.2f s total):\n  int8 GEMM MACs executed  %8.2f M\n  f32 MACs (non-GEMM)      %8.2f M\n  weights read by the GEMM %8.2f MB\n",
           sec, t8 / sec / 1e6, tf / sec / 1e6, wb_tot / sec / 1e6);
    for (int q = 0; q < ITOFS_OPC_N; q++)
        printf("  %-32s %10.0f /s   (assumed %g cycles each)\n", opn[q], (double)tot[q] / sec, cyc[q]);
    const double cy = est_cycles(tot, tf) / sec;
    printf("  ESTIMATED non-GEMM work: %.1f M cycles per second of audio = %.2f of one 240 MHz core (float work runs on core 0 only)\n",
           cy / 1e6, cy / 240e6);
    return 0;
}
