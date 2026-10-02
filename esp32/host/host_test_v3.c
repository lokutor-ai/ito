// Host verification of the ItoFS C engine for arch-3 (v3) blobs against the engine-numerics PyTorch reference
// (golden/ref<k>_w8a<8|16>.bin, written by Lokutor's exporter).
//
//   cc -std=c99 -O2 -ffp-contract=off -o host_test_v3 engine/itofs.c engine/host_test_v3.c -lm
//   ./host_test_v3 <itofs_weights.bin> <golden dir> [--act 8|16] [--wav]
//
// Per golden sentence:
//   A  free-running vs the reference (same tokens / style / source noise / phase0): durations, per-stage SNR
//   B  duration head given the reference's encoder output h
//   C  stage-isolated: prosody given ref h + durations; mel head given ref pred; vocoder given ref mel (+ F0)
//   D  streaming: chunks of 1, 2, 3, 5, 8, 16, 32 frames vs one whole-utterance chunk, BIT-IDENTICAL (golden and seeded noise),
//      each also with the row-parallel (dual-core) split and with a 2-frame first chunk
//   E  (--wav) C audio, free-running and with the float model's durations
//   F  executed operations before the first audio chunk and per second of audio
#include "itofs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { char name[64]; int dt; int n; void *data; } rec_t;
typedef struct { rec_t r[64]; int n; } refs_t;

static int load_refs(const char *path, refs_t *R)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char mg[4];
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "ITFR", 4)) { fclose(f); return -2; }
    R->n = 0;
    for (;;) {
        uint32_t nl;
        if (fread(&nl, 4, 1, f) != 1) break;
        rec_t *r = &R->r[R->n++];
        if (nl >= sizeof r->name || fread(r->name, 1, nl, f) != nl) { fclose(f); return -3; }
        r->name[nl] = 0;
        uint32_t dt, cnt;
        if (fread(&dt, 4, 1, f) != 1 || fread(&cnt, 4, 1, f) != 1) { fclose(f); return -3; }
        r->dt = (int)dt; r->n = (int)cnt;
        size_t es = dt == 2 ? 8 : 4;
        r->data = malloc(es * cnt + 16);
        if (fread(r->data, es, cnt, f) != cnt) { fclose(f); return -4; }
    }
    fclose(f);
    return 0;
}

static rec_t *get(refs_t *R, const char *name)
{
    for (int i = 0; i < R->n; i++) if (!strcmp(R->r[i].name, name)) return &R->r[i];
    fprintf(stderr, "missing record %s\n", name);
    exit(1);
}

static double snr_db(const float *ref, const float *x, long n)
{
    double s = 0, e = 0;
    for (long i = 0; i < n; i++) { double d = (double)ref[i] - x[i]; s += (double)ref[i] * ref[i]; e += d * d; }
    return e == 0 ? 999.0 : 10 * log10(s / e);
}

typedef struct { int T, hop, NF, NM; float *pred, *src, *hfeat, *spec, *mel, *f0; } taps_t;
static void tap_fn(void *u, int what, int t, const float *row, int w)
{
    taps_t *tp = u;
    if (t >= tp->T) return;
    if (what == ITOFS_TAP_PRED) memcpy(tp->pred + (size_t)t * 3, row, 12);
    else if (what == ITOFS_TAP_SRC) memcpy(tp->src + (size_t)t * tp->hop, row, (size_t)w * 4);
    else if (what == ITOFS_TAP_HFEAT) memcpy(tp->hfeat + (size_t)t * tp->NF, row, (size_t)w * 4);
    else if (what == ITOFS_TAP_SPEC) memcpy(tp->spec + (size_t)t * tp->NF, row, (size_t)w * 4);
    else if (what == ITOFS_TAP_MEL) memcpy(tp->mel + (size_t)t * tp->NM, row, (size_t)w * 4);
    else if (what == ITOFS_TAP_F0) tp->f0[t] = row[0];
}

static int g_act = 8;
static itofs_ctx_t *make_ctx_w(const itofs_model_t *m, int max_tokens, int chunk, int whole)
{
    itofs_limits_t lim = { max_tokens, chunk, g_act, whole };
    size_t hb, bb;
    itofs_arena_bytes(m, &lim, &hb, &bb);
    itofs_ctx_t *c = malloc(sizeof *c);
    void *hot = malloc(hb), *bulk = malloc(bb);
    int e = itofs_init(c, m, &lim, hot, hb, bulk, bb);
    if (e) { fprintf(stderr, "init: %s\n", itofs_strerror(e)); exit(1); }
    return c;
}
static itofs_ctx_t *make_ctx(const itofs_model_t *m, int max_tokens, int chunk) { return make_ctx_w(m, max_tokens, chunk, 0); }

static int g_first = 0;      // > 0: the first chunk has this many frames (the firmware's low-latency first chunk)
static long synth(itofs_ctx_t *c, const int *tok, int n, int style, uint32_t seed, float *out, long cap)
{
    int T = itofs_begin(c, tok, n, style, seed);
    if (T < 0) { fprintf(stderr, "begin: %s\n", itofs_strerror(T)); exit(1); }
    long pos = 0;
    for (;;) {
        const long lim = (pos == 0 && g_first > 0) ? (long)g_first * c->m->hop : cap - pos;
        int got = itofs_next_chunk_f32(c, out + pos, (int)(lim < cap - pos ? lim : cap - pos));
        if (got < 0) { fprintf(stderr, "next_chunk: %s\n", itofs_strerror(got)); exit(1); }
        if (got == 0) break;
        pos += got;
    }
    return pos;
}

static void write_wav(const char *path, const float *y, long n, int sr)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    uint32_t data = (uint32_t)n * 2, riff = 36 + data, fmt = 16, rate = (uint32_t)sr, br = (uint32_t)sr * 2;
    uint16_t pcm = 1, ch = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f);
    fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (long i = 0; i < n; i++) {
        float v = y[i] * 32767.f;
        v = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v);
        int16_t s = (int16_t)lrintf(v);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

static const char *gname[ITOFS_G_N] = {"text side", "prosody net", "harmonic STFT proj", "vocoder embed",
                                      "ConvNeXt blocks", "head (out linear)", "mel head"};

static void zero_macs(itofs_ctx_t *c)
{
    memset(c->macs, 0, sizeof c->macs); memset(c->macs_f32, 0, sizeof c->macs_f32); memset(c->macs_exec, 0, sizeof c->macs_exec);
}

// row-parallel hook as the chip's would run it, but on one thread: the second core's part first, an uneven split
static void host_par(void *user, itofs_body_fn body, void *arg, int n)
{
    (void)user;
    const int h = n / 3;
    body(arg, h, n, 1);
    if (h > 0) body(arg, 0, h, 0);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <blob> <golden dir> [--act 8|16] [--wav]\n", argv[0]); return 2; }
    int do_wav = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--wav")) do_wav = 1;
        else if (!strcmp(argv[i], "--act") && i + 1 < argc) g_act = atoi(argv[++i]);
    }
    if (g_act != 8 && g_act != 16) { fprintf(stderr, "--act must be 8 or 16\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long bs = ftell(f); fseek(f, 0, SEEK_SET);
    void *blob = NULL;
    if (posix_memalign(&blob, 64, (size_t)bs)) return 1;
    if (fread(blob, 1, (size_t)bs, f) != (size_t)bs) return 1;
    fclose(f);
    itofs_model_t m;
    int e = itofs_model_init(&m, blob, (size_t)bs);
    if (e) { fprintf(stderr, "model init: %s\n", itofs_strerror(e)); return 1; }
    if (m.arch != 3) { fprintf(stderr, "not an arch-3 blob (this test covers arch-3 blobs)\n"); return 2; }
    printf("act_bits %d (int8-weight layers: %s; int16-weight layers: always 15-bit)\n", g_act, g_act == 8 ? "1 int8 plane" : "2 planes");
    printf("blob %s: %ld bytes, %ld int8 + %ld int16 + %ld f32 params | text %d x%d, GRU %d%s, pros %d x%d, mel head %d x%d -> %d, "
           "vocoder %d/%d x%d, n_fft %d hop %d sr %d (%d fps), %d styles\n", argv[1], bs, m.n_int8_params, m.n_int16_params,
           m.n_f32_params, m.text_dim, m.text_layers, m.rnn_hidden, m.rnn_bidir ? " bidirectional" : " forward", m.pros_dim,
           m.pros_layers, m.mel_dim, m.mel_layers, m.n_mels, m.dec_dim, m.dec_inter, m.dec_blocks, m.n_fft, m.hop, m.sr, m.fps, m.n_styles);
    const int chunk = 2400 / m.hop > 0 ? 2400 / m.hop : 1;
    {
        itofs_limits_t lim = { 400, chunk, g_act, 0 };
        size_t hb, bb; itofs_arena_bytes(&m, &lim, &hb, &bb);
        printf("arenas (max_tokens 400, chunk %d frames): hot %zu bytes, bulk %zu bytes\n", chunk, hb, bb);
    }
    int fails = 0;
    double mac_tot[ITOFS_G_N] = {0}, macf_tot[ITOFS_G_N] = {0}, mace_tot[ITOFS_G_N] = {0}, sec_tot = 0, tok_tot = 0;
    double worst_free_spec = 999, worst_iso_spec = 999, worst_iso_mel = 999;
    for (int k = 0; k < 16; k++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/ref%d_w8a%d.bin", argv[2], k, g_act);
        refs_t R;
        if (load_refs(path, &R)) break;
        rec_t *rt = get(&R, "tokens");
        const int n = rt->n, *tok = rt->data;
        const int style = ((int *)get(&R, "style_idx")->data)[0];
        const double ph0 = ((double *)get(&R, "phase0_f64")->data)[0];
        const float *noise = get(&R, "noise")->data;
        const int *ref_dur = get(&R, "np_dur")->data, *t32_dur = get(&R, "t32_dur")->data;
        rec_t *r_audio = get(&R, "np_audio");
        const int T = r_audio->n / m.hop, N = m.n_fft, NM = m.n_mels;
        printf("\n=== golden %d: %d tokens, style %d, T = %d frames (%.2f s)\n", k, n, style, T, (double)T / m.fps);
        taps_t tp = { T, m.hop, N + 2, NM, malloc((size_t)T * 12 + 64), malloc((size_t)T * m.hop * 4 + 64),
                      malloc((size_t)T * (N + 2) * 4 + 64), malloc((size_t)T * (N + 2) * 4 + 64),
                      malloc((size_t)T * NM * 4 + 64), malloc((size_t)T * 4 + 64) };
        const long cap = (long)(T + 64) * m.hop * 2;
        float *y = malloc((size_t)cap * 4), *y2 = malloc((size_t)cap * 4);
        itofs_ctx_t *c = make_ctx(&m, 400, chunk);
        c->tap = tap_fn; c->tap_user = &tp;
        c->ext_noise = noise; c->ext_phase0 = ph0; c->use_ext_phase0 = 1;
        const float *Rh = get(&R, "np_h")->data, *Rlogd = get(&R, "np_logd")->data, *Rpred = get(&R, "np_pred")->data;
        const float *Rf0 = get(&R, "np_f0")->data, *Rmel = get(&R, "np_mel")->data, *Rsrc = get(&R, "np_src")->data;
        const float *Rhf = get(&R, "np_hfeat")->data, *Rspec = get(&R, "np_spec")->data;

        // ---- A: free-running
        zero_macs(c);
        clock_t c0 = clock();
        itofs_begin(c, tok, n, style, 0);
        clock_t c1 = clock();
        long ns = 0;
        for (;;) {
            int got = itofs_next_chunk_f32(c, y + ns, (int)(cap - ns));
            if (got <= 0) break;
            ns += got;
        }
        clock_t c2 = clock();
        int dur_exact = c->T == T, ndiff = 0;
        for (int i = 0; i < n; i++) ndiff += c->dur[i] != ref_dur[i];
        dur_exact &= ndiff == 0;
        printf("A free-running vs reference a%d: durations %s (%d differ) | h %.1f | %s %.1f | logd %.1f", g_act, dur_exact ? "EXACT" : "DIFFER",
               ndiff, snr_db(Rh, c->h, (long)n * m.text_dim), m.has_sp ? "predicted style s" : "style s",
               snr_db(get(&R, "np_s")->data, c->s, m.style_dim), snr_db(Rlogd, c->logd, n));
        if (dur_exact) {
            const double sp = snr_db(Rspec, tp.spec, (long)T * (N + 2));
            if (sp < worst_free_spec) worst_free_spec = sp;
            printf(" | pred %.1f | F0 %.1f | mel %.1f | src %.1f | harm STFT %.1f | spec %.1f | audio %.1f dB\n",
                   snr_db(Rpred, tp.pred, (long)T * 3), snr_db(Rf0, tp.f0, T), snr_db(Rmel, tp.mel, (long)T * NM),
                   snr_db(Rsrc, tp.src, (long)T * m.hop), snr_db(Rhf, tp.hfeat, (long)T * (N + 2)), sp, snr_db(r_audio->data, y, (long)T * m.hop));
        } else printf("\n");
        if (!dur_exact) fails++;
        for (int g = 0; g < ITOFS_G_N; g++) { mac_tot[g] += c->macs[g]; macf_tot[g] += c->macs_f32[g]; mace_tot[g] += c->macs_exec[g]; }
        sec_tot += (double)c->T / m.fps; tok_tot += n;
        printf("  host speed (1 thread, plain C GEMM; not a chip number): text side %.1f ms, total %.1f ms for %.2f s audio\n",
               1e3 * (c1 - c0) / CLOCKS_PER_SEC, 1e3 * (c2 - c0) / CLOCKS_PER_SEC, (double)ns / m.sr);
        if (do_wav) { snprintf(path, sizeof path, "%s/g%d_c_engine_a%d.wav", argv[2], k, g_act); write_wav(path, y, ns, m.sr); }

        // ---- B: duration head given the reference h
        c->force_h = Rh;
        itofs_begin(c, tok, n, style, 0);
        int ok = 1; for (int i = 0; i < n; i++) ok &= c->dur[i] == ref_dur[i];
        printf("B duration head given ref h: logd %.1f dB, durations %s\n", snr_db(Rlogd, c->logd, n), ok ? "EXACT" : "DIFFER");
        if (!ok) fails++;

        // ---- C: stage-isolated
        c->force_dur = ref_dur;
        synth(c, tok, n, style, 0, y2, cap);
        printf("C prosody given ref h + durations: pred %.1f dB | F0 %.1f dB\n", snr_db(Rpred, tp.pred, (long)T * 3), snr_db(Rf0, tp.f0, T));
        c->force_pred = Rpred;
        synth(c, tok, n, style, 0, y2, cap);
        const double ms = snr_db(Rmel, tp.mel, (long)T * NM);
        if (ms < worst_iso_mel) worst_iso_mel = ms;
        printf("C mel head given ref h + durations + pred: mel %.1f dB | F0 %.1f dB\n", ms, snr_db(Rf0, tp.f0, T));
        c->force_mel = Rmel;
        long nsc = synth(c, tok, n, style, 0, y2, cap);
        printf("C vocoder given ref mel + pred: src %.1f | harm STFT %.1f | spec %.1f | audio %.1f dB\n",
               snr_db(Rsrc, tp.src, (long)T * m.hop), snr_db(Rhf, tp.hfeat, (long)T * (N + 2)),
               snr_db(Rspec, tp.spec, (long)T * (N + 2)), snr_db(r_audio->data, y2, nsc));
        c->force_f0 = Rf0;
        long nsf = synth(c, tok, n, style, 0, y2, cap);
        const double sp2 = snr_db(Rspec, tp.spec, (long)T * (N + 2));
        if (sp2 < worst_iso_spec) worst_iso_spec = sp2;
        printf("C vocoder given ref mel + pred + F0: src %.1f | spec %.1f | audio %.1f dB\n",
               snr_db(Rsrc, tp.src, (long)T * m.hop), sp2, snr_db(r_audio->data, y2, nsf));
        c->force_f0 = NULL; c->force_mel = NULL; c->force_pred = NULL; c->force_h = NULL; c->force_dur = NULL;

        // ---- D: streaming vs whole, bit-identical
        c->tap = NULL;
        int bigc = 1; while (bigc < T + 8) bigc <<= 1;
        itofs_ctx_t *cw = make_ctx(&m, 400, bigc);
        for (int mode = 0; mode < 2; mode++) {       // 0: external noise/phase0 (golden), 1: seeded generator
            cw->ext_noise = mode ? NULL : noise; cw->ext_phase0 = ph0; cw->use_ext_phase0 = !mode;
            long nw = synth(cw, tok, n, style, 12345u, y, cap);
            printf("D streaming vs whole (%s): whole %ld samples;", mode ? "seeded noise, seed 12345" : "golden noise", nw);
            const int chunks[] = {1, 2, 3, 5, 8, 16, 24, 32};
            for (int ci = 0; ci < 7; ci++) {
                itofs_ctx_t *cs = make_ctx(&m, 400, chunks[ci]);
                cs->ext_noise = cw->ext_noise; cs->ext_phase0 = ph0; cs->use_ext_phase0 = !mode;
                long nsx = synth(cs, tok, n, style, 12345u, y2, cap);
                int same = nsx == nw && !memcmp(y, y2, (size_t)nw * 4);
                cs->par = host_par;                  // the dual-core row split must not change a bit either
                nsx = synth(cs, tok, n, style, 12345u, y2, cap);
                const int same_p = nsx == nw && !memcmp(y, y2, (size_t)nw * 4);
                g_first = 2;                         // 2-frame first chunk, then chunks[ci] (row-parallel)
                nsx = synth(cs, tok, n, style, 12345u, y2, cap);
                g_first = 0;
                const int same_f = nsx == nw && !memcmp(y, y2, (size_t)nw * 4);
                printf(" %d:%s%s%s", chunks[ci], same ? "identical" : "DIFF", same_p ? "" : "(row-parallel DIFF)", same_f ? "" : "(2-frame first chunk DIFF)");
                if (!same || !same_p || !same_f) fails++;
            }
            printf("\n");
        }

        // ---- F: executed operations before the first chunk
        {
            itofs_ctx_t *cf = make_ctx(&m, 400, chunk);
            cf->ext_noise = noise; cf->ext_phase0 = ph0; cf->use_ext_phase0 = 1;
            zero_macs(cf);
            itofs_begin(cf, tok, n, style, 0);
            double tx8 = cf->macs_exec[ITOFS_G_TEXT], txf = cf->macs_f32[ITOFS_G_TEXT];
            itofs_next_chunk_f32(cf, y2, (int)cap);
            double fr8 = 0, frf = 0;
            for (int g = 0; g < ITOFS_G_N; g++) if (g != ITOFS_G_TEXT) { fr8 += cf->macs_exec[g]; frf += cf->macs_f32[g]; }
            printf("F first chunk (%d frames): text side %.1f M int8 + %.2f M f32 MACs (%d tokens) | frame pipeline fill %.1f M int8 + "
                   "%.2f M f32 MACs | total %.1f M int8 + %.2f M f32\n", chunk, tx8 / 1e6, txf / 1e6, n, fr8 / 1e6, frf / 1e6,
                   (tx8 + fr8) / 1e6, (txf + frf) / 1e6);
        }

        // ---- G: incremental text side (forward-GRU models) vs whole-sentence text side: bit-identical; first-audio ops
        if (!m.rnn_hidden || !m.rnn_bidir) {
            itofs_ctx_t *ci = make_ctx(&m, 400, chunk), *cw2 = make_ctx_w(&m, 400, chunk, 1);
            cw2->force_whole_text = 1;
            for (int q = 0; q < 2; q++) {
                itofs_ctx_t *cc = q ? cw2 : ci;
                cc->ext_noise = noise; cc->ext_phase0 = ph0; cc->use_ext_phase0 = 1;
            }
            zero_macs(ci);
            itofs_begin(ci, tok, n, style, 0);
            long pi = itofs_next_chunk_f32(ci, y, (int)cap);
            double i8 = 0, i32 = 0;
            for (int g = 0; g < ITOFS_G_N; g++) { i8 += ci->macs_exec[g]; i32 += ci->macs_f32[g]; }
            const int tok_first = ci->n_done;
            for (;;) { int gg = itofs_next_chunk_f32(ci, y + pi, (int)(cap - pi)); if (gg <= 0) break; pi += gg; }
            long pw = synth(cw2, tok, n, style, 0, y2, cap);
            const int same_pcm = pi == pw && !memcmp(y, y2, (size_t)pw * 4);
            const int same_txt = !memcmp(ci->h, cw2->h, (size_t)n * m.text_dim * 4) && !memcmp(ci->logd, cw2->logd, (size_t)n * 4) &&
                                 !memcmp(ci->dur, cw2->dur, (size_t)n * 4);
            printf("G incremental vs whole-sentence text side: h/logd/durations %s | PCM %s | before the first chunk: %.1f M int8 + "
                   "%.2f M f32 MACs (%d of %d tokens final)\n", same_txt ? "bit-identical" : "DIFFER", same_pcm ? "bit-identical" : "DIFFER",
                   i8 / 1e6, i32 / 1e6, tok_first, n);
            if (!same_pcm || !same_txt) fails++;
        }

        // ---- E: audio with the float model's durations
        if (do_wav) {
            c->force_dur = t32_dur;
            long nq = synth(c, tok, n, style, 0, y, cap);
            snprintf(path, sizeof path, "%s/g%d_c_engine_a%d_floatdur.wav", argv[2], k, g_act); write_wav(path, y, nq, m.sr);
            rec_t *tf = get(&R, "nptf_audio");
            printf("E with the float durations: C vs reference audio %.1f dB (wav written)\n", snr_db(tf->data, y, nq < tf->n ? nq : tf->n));
            c->force_dur = NULL;
        }
        free(tp.pred); free(tp.src); free(tp.hfeat); free(tp.spec); free(tp.mel); free(tp.f0); free(y); free(y2);
    }
    printf("\nMACs per second of audio (C graph, %.2f s over the golden sentences). logical = weights x rows; executed = int8 x int8\n"
           "MACs the GEMM kernel runs (x activation planes x weight planes); f32 = float multiply-adds outside the GEMMs:\n", sec_tot);
    double fr = 0, frf = 0, fre = 0;
    for (int g = 0; g < ITOFS_G_N; g++) {
        if (g == ITOFS_G_TEXT)
            printf("  %-22s %8.3f M logical, %8.3f M executed int8, %6.3f M f32 per token\n", gname[g],
                   mac_tot[g] / tok_tot / 1e6, mace_tot[g] / tok_tot / 1e6, macf_tot[g] / tok_tot / 1e6);
        else {
            printf("  %-22s %8.2f M logical, %8.2f M executed int8, %6.2f M f32 per second\n", gname[g], mac_tot[g] / sec_tot / 1e6,
                   mace_tot[g] / sec_tot / 1e6, macf_tot[g] / sec_tot / 1e6);
            fr += mac_tot[g]; frf += macf_tot[g]; fre += mace_tot[g];
        }
    }
    printf("  %-22s %8.2f M logical, %8.2f M executed int8, %6.2f M f32 per second (frame rate, excluding the text side)\n",
           "TOTAL", fr / sec_tot / 1e6, fre / sec_tot / 1e6, frf / sec_tot / 1e6);
    printf("\nworst SNR: free-running spec %.1f dB | mel head given ref pred %.1f dB | vocoder spec given ref mel + F0 %.1f dB\n",
           worst_free_spec, worst_iso_mel, worst_iso_spec);
    printf("%s (%d failures: durations / streaming bit-exactness)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
