// Ito on the ESP32-S3: StyleTTS 2 token ids in, 24 kHz audio out over I2S. Runs Ito v3 blobs (arch 3; the engine also
// still reads the older arch-2 format).
//
//   boot: load the weight blob (flash partition "weights") into PSRAM (or run it memory-mapped from flash when it does
//         not fit), run the self-test stored in the blob (the host C engine's PCM hash for golden sentence 0), run the
//         board benchmark (hardware build only: kernels + the demo sentences in every weight-staging mode, then keep the
//         fastest mode), then speak the demo sentences (stored in the blob; main/demo_sentences.h is the fallback).
//   BOOT button (GPIO0): speak the demo sentences again.
//   serial (UART0, 115200): "say 0,72,156,..."  speak token ids (tools/say.py)
//                           "style <0-63>" | "act <8|16>" | "demo" | "demo <k>" | "test" | "stats" | "bench" |
//                           "wmode <0|1|2>" (weight staging: direct / copy / gdma) | "first <n>" (frames in the
//                           low-latency first chunk, default 2 = 25 ms) | "help"
//   I2S: BCLK GPIO15, LRCLK GPIO16, DOUT GPIO17, sample rate from the blob (24 kHz), 16-bit, mono duplicated to stereo.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_random.h"
#include "esp_cache.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#ifndef ITOFS_QEMU
#include "driver/i2s_std.h"
#endif
#include "itofs.h"
#include "kernels_s3.h"
#include "demo_sentences.h"

#define PIN_BCLK 15
#define PIN_WS 16
#define PIN_DOUT 17
#define PIN_BOOT 0
#define MAX_TOKENS 400
#define CHUNK_TARGET 2400            // samples per chunk (100 ms at 24 kHz): chunk_frames = 2400 / hop
#define NBUF 2                       // double buffer between synthesis and I2S
#ifndef FIRST_FRAMES
#define FIRST_FRAMES 2               // low-latency first chunk: 2 frames (25 ms at 80 fps), then CHUNK_TARGET chunks. The output
                                     // is bit-identical to any other chunking; `first <n>` changes it (n >= chunk = off)
#endif
#ifndef ITOFS_ACT_BITS
#define ITOFS_ACT_BITS 8             // activations of the int8-weight layers: 8 (one int8 plane, default) or 16; `act` command
#endif

static itofs_model_t s_model;
static itofs_ctx_t s_ctx;
static int s_hop, s_sr, s_chunk_frames, s_chunk_samples, s_style, s_first = FIRST_FRAMES;
static const int32_t *s_st, *s_st16; // self-test records from the blob ("selftest": default 8-bit activations,
                                     // "selftest_a16": 16-bit); NULL if absent

typedef struct { int16_t *pcm; int n; } buf_t;      // n = samples, 0 = end of utterance
static QueueHandle_t s_free_q, s_full_q, s_req_q, s_done_q;
static volatile int s_underruns, s_playing;

typedef struct { int kind; int n; int16_t tok[MAX_TOKENS]; } req_t;   // 0 say, 1 demos, 2 self-test, 3 style, 4 act bits, 5 bench, 6 wmode, 7 icprof, 8 first
static req_t s_req_tmp;

static double ms(int64_t us) { return us / 1000.0; }

static void mem_report(const char *tag)
{
    size_t pt = heap_caps_get_total_size(MALLOC_CAP_SPIRAM), pmin = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    size_t it = heap_caps_get_total_size(MALLOC_CAP_INTERNAL), imin = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    printf("MEM %s: peak PSRAM used %u / %u KB, peak internal SRAM used %u / %u KB\n", tag,
           (unsigned)((pt - pmin) / 1024), (unsigned)(pt / 1024), (unsigned)((it - imin) / 1024), (unsigned)(it / 1024));
}

// ------------------------------------------------------------------------------------------------------------------
// audio output
// ------------------------------------------------------------------------------------------------------------------
#ifndef ITOFS_QEMU
static i2s_chan_handle_t s_tx;
static void i2s_setup(void)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 6;
    cc.dma_frame_num = 480;
    cc.auto_clear = true;            // silence (not a repeated buffer) on underrun
    ESP_ERROR_CHECK(i2s_new_channel(&cc, &s_tx, NULL));
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(24000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = { .mclk = I2S_GPIO_UNUSED, .bclk = PIN_BCLK, .ws = PIN_WS, .dout = PIN_DOUT, .din = I2S_GPIO_UNUSED,
                      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false } },
    };
    sc.clk_cfg.sample_rate_hz = (uint32_t)s_sr;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx, &sc));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx));
}
#endif

static void play_task(void *arg)
{
    (void)arg;
    int16_t *st = heap_caps_malloc(sizeof(int16_t) * 2 * (size_t)s_chunk_samples + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    buf_t b;
    for (;;) {
        if (s_playing && xQueueReceive(s_full_q, &b, 0) != pdTRUE) {
            s_underruns++;           // synthesis did not keep up: the DAC plays silence meanwhile
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        } else if (!s_playing) {
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        }
        if (b.n == 0) { s_playing = 0; xQueueSend(s_done_q, &b, portMAX_DELAY); continue; }
        s_playing = 1;
        for (int i = 0; i < b.n; i++) { st[2 * i] = b.pcm[i]; st[2 * i + 1] = b.pcm[i]; }
        xQueueSend(s_free_q, &b, portMAX_DELAY);
#ifndef ITOFS_QEMU
        size_t wr = 0;
        i2s_channel_write(s_tx, st, sizeof(int16_t) * 2 * (size_t)b.n, &wr, portMAX_DELAY);
#else
        (void)st;
#endif
    }
}

// ------------------------------------------------------------------------------------------------------------------
// synthesis
// ------------------------------------------------------------------------------------------------------------------
typedef struct { double ttfa_ms, begin_ms, compute_ms, audio_s; int frames, tok_first; } stats_t;

// Synthesise tokens; stream chunks to the player (play=1) or into out[] (play=0).
static int speak(const int *tok, int n, int style, uint32_t seed, int play, int16_t *out, long out_max, stats_t *S)
{
    S->tok_first = 0;
    int64_t t0 = esp_timer_get_time();
    int T = itofs_begin(&s_ctx, tok, n, style, seed);
    int64_t t1 = esp_timer_get_time();
    if (T < 0) { printf("ERROR itofs_begin: %s\n", itofs_strerror(T)); return T; }
    int64_t compute = t1 - t0, ttfa = -1;
    long pos = 0;
    s_underruns = 0;
    for (;;) {
        buf_t b = { NULL, 0 };
        if (play) xQueueReceive(s_free_q, &b, portMAX_DELAY);
        int16_t *dst = play ? b.pcm : out + pos;
        int room = play ? s_chunk_samples : (int)((out_max - pos) < s_chunk_samples ? (out_max - pos) : s_chunk_samples);
        if (pos == 0 && s_first > 0 && s_first * s_hop < room) room = s_first * s_hop;     // low-latency first chunk
        int64_t a = esp_timer_get_time();
        int got = room >= s_hop ? itofs_next_chunk(&s_ctx, dst, room) : 0;
        int64_t e = esp_timer_get_time();
        compute += e - a;
        if (got <= 0) { if (play) xQueueSend(s_free_q, &b, portMAX_DELAY); break; }
        if (ttfa < 0) { ttfa = e - t0; S->tok_first = s_ctx.n_done; }
        pos += got;
        if (play) { b.n = got; xQueueSend(s_full_q, &b, portMAX_DELAY); }
    }
    if (play) {
        buf_t end = { NULL, 0 };
        xQueueSend(s_full_q, &end, portMAX_DELAY);
        xQueueReceive(s_done_q, &end, portMAX_DELAY);
    }
    S->begin_ms = ms(t1 - t0); S->ttfa_ms = ms(ttfa); S->compute_ms = ms(compute);
    S->frames = s_ctx.T; S->audio_s = (double)pos / s_sr; (void)T;
    return (int)pos;
}

static void print_timing(const char *what, const stats_t *S, int n_tok, int style)
{
    printf("TIMING %s: %d tokens, style %d, %.2f s audio | first chunk %.1f ms (begin %.1f ms; %s text side, %d tokens final at first chunk) | "
           "compute %.1f ms, RTF %.3f | underruns %d\n",
           what, n_tok, style, S->audio_s, S->ttfa_ms, S->begin_ms, s_ctx.text_incr ? "incremental" : "whole-sentence", S->tok_first,
           S->compute_ms, S->compute_ms / 1000.0 / S->audio_s, s_underruns);
    mem_report(what);
}

static int s_tok_buf[MAX_TOKENS];

// demo sentences: from the blob ("demos" / "demo_text") or the compiled-in fallback
#define MAX_DEMOS 8
static int s_ndemo, s_demo_len[MAX_DEMOS], s_demo_style[MAX_DEMOS];
static const int32_t *s_demo_tok32[MAX_DEMOS];
static const char *s_demo_txt[MAX_DEMOS];
static int s_demo_from_blob;

static void load_demos(const void *blob, size_t bs)
{
    const void *d; size_t nb; int dt;
    if (!itofs_blob_find(blob, bs, "demos", &d, &nb, &dt) && dt == 2 && nb >= 4) {
        const int32_t *r = d;
        const int n = r[0] < MAX_DEMOS ? r[0] : MAX_DEMOS;
        size_t p = 1;
        for (int k = 0; k < n && (p + 2) * 4 <= nb; k++) {
            s_demo_len[k] = r[p]; s_demo_style[k] = r[p + 1]; s_demo_tok32[k] = r + p + 2;
            p += 2 + (size_t)r[p];
            s_ndemo = k + 1;
        }
        const char *txt = NULL; size_t tl = 0;
        if (!itofs_blob_find(blob, bs, "demo_text", &d, &nb, &dt) && dt == 2) { txt = d; tl = nb; }
        for (int k = 0; k < s_ndemo; k++) {
            s_demo_txt[k] = txt ? txt : "";
            if (txt) { size_t l = strnlen(txt, tl); txt += l + 1; tl -= (l + 1 <= tl ? l + 1 : tl); if (!tl) txt = NULL; }
        }
        s_demo_from_blob = 1;
        return;
    }
    s_ndemo = N_DEMOS;
    for (int k = 0; k < N_DEMOS; k++) { s_demo_len[k] = demo_len[k]; s_demo_style[k] = demo_style[k]; s_demo_txt[k] = demo_text[k]; }
}

static void demo_tokens(int k, int *dst)
{
    for (int i = 0; i < s_demo_len[k]; i++) dst[i] = s_demo_from_blob ? (int)s_demo_tok32[k][i] : (int)demo_tok[k][i];
}

// The blob's self-test record: golden sentence 0 with a fixed seed, and the fnv32 hash of the int16 PCM the HOST C
// engine produced for it. The engine is deterministic IEEE float plus an exact int8 GEMM, so the chip must reproduce
// it bit for bit. `both`: also run with the second core disabled (the dual-core GEMM split must not change a bit).
static void self_test_one(const int32_t *s_st, int both, int dump)
{
    const int n = s_st[1], style = s_st[2], nref = s_st[4], act = s_st[6], act_saved = s_ctx.act_bits;
    const uint32_t seed = (uint32_t)s_st[3], href = (uint32_t)s_st[5];
    s_ctx.act_bits = act;            // the record says which activation precision the host reference used
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * ((size_t)nref + (size_t)s_chunk_samples), MALLOC_CAP_SPIRAM);
    if (!pcm || n > MAX_TOKENS) { printf("SELFTEST ERROR: no memory\n"); return; }
    for (int i = 0; i < n; i++) s_tok_buf[i] = s_st[7 + i];
    for (int pass = 0; pass < (both ? 2 : 1); pass++) {
        s3_dual_enabled = pass == 0;
        stats_t S;
        int got = speak(s_tok_buf, n, style, seed, 0, pcm, nref + s_chunk_samples, &S);
        uint32_t h = 2166136261u;
        for (int i = 0; i < got; i++) { h ^= (uint16_t)pcm[i]; h *= 16777619u; }
        const int ok = got == nref && h == href;
        printf("SELFTEST %s, %d-bit activations: golden sentence 0 vs host C engine: %d samples (host %d), fnv32 %08lx (host %08lx) -> %s\n",
               pass == 0 ? "dual-core" : "single-core", act, got, nref, (unsigned long)h, (unsigned long)href, ok ? "PASS" : "FAIL");
        if (pass == 0) print_timing("selftest(no playback)", &S, n, style);
#ifdef ITOFS_QEMU
        if (pass == 0 && dump) {     // full PCM for the sample-by-sample comparison with the host file
            for (int i = 0; i < got; i += 32) {
                printf("PCM %d", i);
                for (int j = i; j < i + 32 && j < got; j++) printf(" %04x", (uint16_t)pcm[j]);
                printf("\n");
            }
            printf("PCMEND %d\n", got);
        }
#endif
    }
    s3_dual_enabled = 1;
    s_ctx.act_bits = act_saved;
    heap_caps_free(pcm);
}

static void self_test(int both)
{
    if (!s_st) { printf("SELFTEST skipped: this weight blob has no self-test record\n"); return; }
    self_test_one(s_st, both, 1);
    if (both && s_st16) self_test_one(s_st16, 0, 0);
}

static void play_demos(int only)
{
    for (int k = 0; k < s_ndemo; k++) {
        if (only >= 0 && k != only) continue;
        demo_tokens(k, s_tok_buf);
        printf("SAY demo %d: \"%s\"\n", k, s_demo_txt[k]);
        stats_t S;
        speak(s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, 1, NULL, 0, &S);
        print_timing("demo", &S, s_demo_len[k], s_demo_style[k]);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// board benchmark: the demo sentences without playback, in every available weight-staging mode. Per sentence: time of
// the text side, time to the first 100 ms chunk (= TTFA before I2S), total compute, RTF, and the executed operations
// (int8 MACs the GEMM runs + f32 multiply-adds) with the time spent inside the GEMM, so effective GOPS can be read off.
// Ends with BOARD_SUMMARY (one line) and keeps the fastest mode.
// ------------------------------------------------------------------------------------------------------------------
static double ctx_ops(int which)        // which 0: int8 executed, 1: f32
{
    double a = 0;
    for (int g = 0; g < ITOFS_G_N; g++) a += which ? s_ctx.macs_f32[g] : s_ctx.macs_exec[g];
    return a;
}

static void tts_bench(void)
{
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_INTERNAL);   // one chunk, overwritten
    if (!pcm) { printf("BENCH ERROR: no memory for the PCM buffer\n"); return; }
    double best_rtf = 1e30, sum_ttfa[3] = {0}, sum_rtf[3] = {0}, gops[3] = {0};
    int best = s3_wmode, nok[3] = {0};
    double tt[3][MAX_DEMOS] = {{0}}, tok_n[MAX_DEMOS] = {0};
    const int saved = s3_wmode;
    for (int mode = 0; mode < 3; mode++) {
        if (s3_set_wmode(mode)) { printf("BENCH_TTS wmode %s: unavailable\n", s3_wmode_name(mode)); continue; }
        double gm = 0, gu = 0;
        for (int k = 0; k < s_ndemo; k++) {
            demo_tokens(k, s_tok_buf);
            const int n = s_demo_len[k];
            const double o8a = ctx_ops(0), ofa = ctx_ops(1);
            s3_stats_reset();
            const int64_t t0 = esp_timer_get_time();
            int T = itofs_begin(&s_ctx, s_tok_buf, n, s_demo_style[k], 1u + (uint32_t)k);
            const int64_t t1 = esp_timer_get_time();
            if (T < 0) { printf("BENCH ERROR begin: %s\n", itofs_strerror(T)); break; }
            long pos = 0;
            int got = itofs_next_chunk(&s_ctx, pcm, s_first > 0 && s_first < s_chunk_frames ? s_first * s_hop : s_chunk_samples);
            const int64_t t2 = esp_timer_get_time();
            const double o8f = ctx_ops(0) - o8a, off = ctx_ops(1) - ofa, gus_first = (double)s3_stats.us;
            while (got > 0) { pos += got; got = itofs_next_chunk(&s_ctx, pcm, s_chunk_samples); }
            const int64_t t3 = esp_timer_get_time();
            const double o8 = ctx_ops(0) - o8a, of = ctx_ops(1) - ofa, aud = (double)pos / s_sr, tot = (double)(t3 - t0);
            const double rtf = tot / 1e6 / aud;
            printf("BENCH_TTS wmode %s demo %d: %d tokens, %.2f s audio | text side %.1f ms | first chunk %.1f ms "
                   "(%.1f M int8 + %.2f M f32 ops, GEMM %.1f ms) | total %.1f ms, RTF %.3f | ops %.1f M int8 + %.1f M f32, "
                   "GEMM %.1f ms = %.3f GMAC/s, non-GEMM %.1f ms | weights streamed %.1f MB\n",
                   s3_wmode_name(mode), k, n, aud, (t1 - t0) / 1e3, (t2 - t0) / 1e3, o8f / 1e6, off / 1e6, gus_first / 1e3,
                   tot / 1e3, rtf, o8 / 1e6, of / 1e6, s3_stats.us / 1e3, s3_stats.macs / (s3_stats.us * 1e3 + 1e-9),
                   (tot - s3_stats.us) / 1e3, s3_stats.wbytes / 1e6);
            sum_ttfa[mode] += (t2 - t0) / 1e3; sum_rtf[mode] += rtf; nok[mode]++;
            tt[mode][k] = (t2 - t0) / 1e3; tok_n[k] = n;
            gm += s3_stats.macs; gu += s3_stats.us;
        }
        gops[mode] = gm / (gu * 1e3 + 1e-9);
        if (nok[mode] && sum_rtf[mode] / nok[mode] < best_rtf) { best_rtf = sum_rtf[mode] / nok[mode]; best = mode; }
        if (nok[mode] >= 2) {     // TTFA = a + b * tokens (least squares over the demos)
            double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = nok[mode];
            for (int k = 0; k < m; k++) { sx += tok_n[k]; sy += tt[mode][k]; sxx += tok_n[k] * tok_n[k]; sxy += tok_n[k] * tt[mode][k]; }
            const double b = (m * sxy - sx * sy) / (m * sxx - sx * sx + 1e-9), a = (sy - b * sx) / m;
            printf("BENCH_TTFA_MODEL wmode %s: first chunk = %.1f ms + %.3f ms/token -> under 200 ms up to %d tokens\n",
                   s3_wmode_name(mode), a, b, b > 0 ? (int)((200 - a) / b) : -1);
        }
    }
    s3_set_wmode(nok[best] ? best : saved);
    printf("BOARD_SUMMARY fw=v3bench arch=%d act=%d best_wmode=%s", s_model.arch, s_ctx.act_bits, s3_wmode_name(s3_wmode));
    for (int mode = 0; mode < 3; mode++)
        if (nok[mode]) printf(" | %s: mean first chunk %.1f ms, mean RTF %.3f, GEMM %.3f GMAC/s", s3_wmode_name(mode),
                              sum_ttfa[mode] / nok[mode], sum_rtf[mode] / nok[mode], gops[mode]);
    printf(" | demo tokens");
    for (int k = 0; k < s_ndemo; k++) printf(" %d", s_demo_len[k]);
    printf("\n");
    mem_report("bench");
    heap_caps_free(pcm);
}

#ifdef ITOFS_ICPROF
// ------------------------------------------------------------------------------------------------------------------
// QEMU -icount profile (build with -D ITOFS_ICPROF=1, run QEMU with -icount shift=0): exact instruction counts of the
// demo sentences and the self-test sentence, text in -> first chunk out and the rest, on 1 core, on 2 cores with the
// GEMM halves serialised (exact per-core split) and on 2 cores in parallel. Raw CCOUNT ticks; ICPROF_CALIB converts.
// ------------------------------------------------------------------------------------------------------------------
#include "esp_cpu.h"
static void ic_calib(void)
{
    vTaskDelay(2);
    const uint32_t c0 = esp_cpu_get_cycle_count();
    const int64_t t0 = esp_timer_get_time();
    __asm__ volatile("movi a8, 1000000\n loop a8, 1f\n nop\n nop\n nop\n nop\n1:\n" ::: "a8");
    const uint32_t dc = esp_cpu_get_cycle_count() - c0;
    const int64_t dt = esp_timer_get_time() - t0;
    printf("ICPROF_CALIB 4000002 instructions: ccount %u, esp_timer %lld us\n", (unsigned)dc, (long long)dt);
    double r[8];
    itofs_prof_micro(r);
    printf("ICPROF_MICRO instructions per call (x25 ticks): div %.0f gelu_series %.0f gelu_tail %.0f expf %.0f sqrtf %.0f sincosf %.0f logf %.0f muladd-loop %.0f\n",
           r[0] * 25, r[1] * 25, r[2] * 25, r[3] * 25, r[4] * 25, r[5] * 25, r[6] * 25, r[7] * 25);
}

typedef struct { double el, single, h0, h1, hmax, dtot, wb, f32, i8; long ns, nd, np; } icsnap_t;
static void ic_snap(icsnap_t *s, double el)
{
    s->el = el; s->single = s3_ic.single; s->h0 = s3_ic.h0; s->h1 = s3_ic.h1; s->hmax = s3_ic.hmax; s->dtot = s3_ic.dual_total;
    s->ns = s3_ic.nsingle; s->nd = s3_ic.ndual; s->np = s3_ic.npar; s->wb = s3_stats.wbytes; s->f32 = ctx_ops(1); s->i8 = ctx_ops(0);
}
static void ic_print(const char *what, const icsnap_t *a, const icsnap_t *b)
{
    printf(" | %s el %.0f single %.0f h0 %.0f h1 %.0f hmax %.0f dtot %.0f ns %ld nd %ld np %ld wb %.0f f32 %.0f i8 %.0f", what, b->el - a->el,
           b->single - a->single, b->h0 - a->h0, b->h1 - a->h1, b->hmax - a->hmax, b->dtot - a->dtot, b->ns - a->ns, b->nd - a->nd, b->np - a->np,
           b->wb - a->wb, b->f32 - a->f32, b->i8 - a->i8);
}

static uint32_t ic_clock(void) { return esp_cpu_get_cycle_count(); }
static const char *ic_pn[ITOFS_PROF_N] = {"PIN", "PROS", "CUR", "SRC", "HFT", "EMB", "BLK", "HEAD", "OLA", "MIN", "MEL", "MOUT",
                                         "tENC", "tGRU", "tPROJ", "tDUR", "tDOUT", "qlin", "quant", "gemm",
                                         "gPIN", "gPROS", "gCUR", "gSRC", "gHFT", "gEMB", "gBLK", "gHEAD", "gOLA", "gMIN", "gMEL", "gMOUT",
                                         "gtENC", "gtGRU", "gtPROJ", "gtDUR", "gtDOUT"};
static void ic_prof_print(const char *what, const double *a, const double *b)
{
    printf("ICPROF_STAGES %s", what);
    for (int k = 0; k < ITOFS_PROF_N; k++) printf(" %s %.0f", ic_pn[k], b[k] - (a ? a[k] : 0));
    printf("\n");
}

#define IC_MAXCH 160
static float s_icch[IC_MAXCH][4];    // per chunk: samples, critical-path ticks (mode 1), weight bytes, f32 MACs
static void ic_one(const char *tag, const int *tok, int n, int style, uint32_t seed, int mode, int16_t *pcm, int first)
{
    itofs_prof_clock = ic_clock;
    memset(itofs_prof, 0, sizeof itofs_prof);
    double pf[ITOFS_PROF_N];
    s3_dual_enabled = mode != 0; s3_ic_serial = mode == 1;
    memset(&s3_ic, 0, sizeof s3_ic); s3_stats_reset();
    icsnap_t z, f, e;
    ic_snap(&z, 0);
    vTaskDelay(1);
    const uint32_t c0 = esp_cpu_get_cycle_count();
    int T = itofs_begin(&s_ctx, tok, n, style, seed);
    if (T < 0) { printf("ICPROF ERROR %s\n", itofs_strerror(T)); return; }
    int got = itofs_next_chunk(&s_ctx, pcm, first > 0 && first < s_chunk_frames ? first * s_hop : s_chunk_samples);
    uint32_t cp = esp_cpu_get_cycle_count();
    double el = (double)(uint32_t)(cp - c0);
    ic_snap(&f, el);
    icsnap_t pv = f;
    s_icch[0][0] = got; s_icch[0][1] = (float)(el - (f.h0 + f.h1 - f.hmax)); s_icch[0][2] = (float)f.wb; s_icch[0][3] = (float)f.f32;
    memcpy(pf, itofs_prof, sizeof pf);
    uint32_t h = 2166136261u;
    long pos = 0;
    int nch = 0;
    while (got > 0) {
        for (int i = 0; i < got; i++) { h ^= (uint16_t)pcm[i]; h *= 16777619u; }
        pos += got; nch++;
        got = itofs_next_chunk(&s_ctx, pcm, s_chunk_samples);
        const uint32_t cn = esp_cpu_get_cycle_count();
        const double de = (double)(uint32_t)(cn - cp);
        el += de; cp = cn;
        icsnap_t cu; ic_snap(&cu, el);
        if (got > 0 && nch < IC_MAXCH) {
            s_icch[nch][0] = got;
            s_icch[nch][1] = (float)(de - ((cu.h0 - pv.h0) + (cu.h1 - pv.h1) - (cu.hmax - pv.hmax)));
            s_icch[nch][2] = (float)(cu.wb - pv.wb); s_icch[nch][3] = (float)(cu.f32 - pv.f32);
        }
        pv = cu;
    }
    ic_snap(&e, el);
    printf("ICPROF %s mode %s first %d tokens %d audio %.4f chunks %d fnv %08lx", tag, mode == 0 ? "1core" : mode == 1 ? "2core-serial" : "2core-par",
           first, n, (double)pos / s_sr, nch, (unsigned long)h);
    ic_print("first", &z, &f);
    ic_print("rest", &f, &e);
    printf("\n");
    if (mode == 0) { ic_prof_print("first", NULL, pf); ic_prof_print("rest", pf, itofs_prof); }
    if (mode == 1) {                 // per chunk: samples / critical-path ticks / weight bytes / f32 MACs
        printf("ICPROF_CHUNKS %s first %d:", tag, first);
        for (int k = 0; k < nch && k < IC_MAXCH; k++) printf(" %.0f/%.0f/%.0f/%.0f", s_icch[k][0], s_icch[k][1], s_icch[k][2], s_icch[k][3]);
        printf("\n");
    }
}

static void icprof(void)
{
    ic_calib();
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_INTERNAL);
    if (!pcm) { printf("ICPROF ERROR no memory\n"); return; }
    printf("ICPROF begin: chunk %d frames, %d-bit activations, wmode %s\n", s_chunk_frames, s_ctx.act_bits, s3_wmode_name(s3_wmode));
    // runs: (mode, first-chunk frames): 1 core; 2 cores serialised (exact per-core split) with the old 8-frame and the
    // new 2-frame first chunk; 2 cores in parallel
    static const int runs[][2] = { {0, 8}, {1, 8}, {1, 2}, {1, 1}, {2, 2} };
    for (int ri = 0; ri < 5; ri++) {
        const int mode = runs[ri][0], first = runs[ri][1];
        if (s_st) {
            int n = s_st[1];
            for (int i = 0; i < n; i++) s_tok_buf[i] = s_st[7 + i];
            ic_one("selftest", s_tok_buf, n, s_st[2], (uint32_t)s_st[3], mode, pcm, first);
        }
        for (int k = 0; k < s_ndemo; k++) {
            char tg[16]; snprintf(tg, sizeof tg, "demo%d", k);
            demo_tokens(k, s_tok_buf);
            ic_one(tg, s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, mode, pcm, first);
        }
    }
    s3_dual_enabled = 1; s3_ic_serial = 0;
    heap_caps_free(pcm);
    printf("ICPROF end\n");
}
#endif

static const void *s_blob; static size_t s_blob_size;

static void board_bench(void)
{
    s3_bench(s_blob, s_blob_size);
    tts_bench();
    printf("BENCH end\n");
}

static void synth_task(void *arg)
{
    (void)arg;
    self_test(0);
#ifndef ITOFS_QEMU
    board_bench();                   // hardware: measure, keep the fastest weight-staging mode (QEMU: `bench` command)
#endif
    play_demos(-1);
    printf("READY (BOOT button = demos, serial: say <ids> | style <k> | demo [k] | test | stats | help)\n");
    for (;;) {
        xQueueReceive(s_req_q, &s_req_tmp, portMAX_DELAY);
        if (s_req_tmp.kind == 0) {
            for (int i = 0; i < s_req_tmp.n; i++) s_tok_buf[i] = s_req_tmp.tok[i];
            stats_t S;
            if (speak(s_tok_buf, s_req_tmp.n, s_style, esp_random(), 1, NULL, 0, &S) > 0) print_timing("say", &S, s_req_tmp.n, s_style);
        } else if (s_req_tmp.kind == 1) {
            play_demos(s_req_tmp.n);
        } else if (s_req_tmp.kind == 2) {
            self_test(1);
        } else if (s_req_tmp.kind == 3) {
            s_style = s_req_tmp.n;
            printf("OK style %d\n", s_style);
        } else if (s_req_tmp.kind == 5) {
            board_bench();
#ifdef ITOFS_ICPROF
        } else if (s_req_tmp.kind == 7) {
            icprof();
#endif
        } else if (s_req_tmp.kind == 8) {
            s_first = s_req_tmp.n;
            printf("OK first chunk %d frames%s\n", s_first, s_first >= s_chunk_frames || s_first <= 0 ? " (= a full chunk)" : "");
        } else if (s_req_tmp.kind == 6) {
            if (s3_set_wmode(s_req_tmp.n)) printf("ERROR weight staging mode %d unavailable\n", s_req_tmp.n);
            else printf("OK weight staging %s\n", s3_wmode_name(s3_wmode));
        } else {
            s_ctx.act_bits = s_req_tmp.n;
            printf("OK %d-bit activations for the int8-weight layers\n", s_ctx.act_bits);
        }
        printf("READY\n");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// input: serial commands + BOOT button
// ------------------------------------------------------------------------------------------------------------------
static req_t s_in_req;

static void handle_line(char *line)
{
    while (*line == ' ' || *line == '\t') line++;
    if (!strncmp(line, "say", 3)) {
        char *p = line + 3;
        int n = 0;
        while (*p) {
            while (*p && (*p < '0' || *p > '9')) p++;
            if (!*p) break;
            long v = strtol(p, &p, 10);
            if (v < 0 || v >= s_model.n_vocab) { printf("ERROR token %ld out of range\n", v); return; }
            if (n >= MAX_TOKENS) { printf("ERROR more than %d tokens (split the text into sentences)\n", MAX_TOKENS); return; }
            s_in_req.tok[n++] = (int16_t)v;
        }
        if (n == 0) { printf("ERROR no tokens\n"); return; }
        s_in_req.kind = 0; s_in_req.n = n;
        printf("OK %d tokens queued\n", n);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "style", 5)) {
        int k = atoi(line + 5);
        if (k < 0 || k >= s_model.n_styles) { printf("ERROR style must be 0-%d\n", s_model.n_styles - 1); return; }
        s_in_req.kind = 3; s_in_req.n = k;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "act", 3)) {
        int b = atoi(line + 3);
        if (b != 8 && b != 16) { printf("ERROR act must be 8 or 16\n"); return; }
        s_in_req.kind = 4; s_in_req.n = b;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "demo", 4)) {
        s_in_req.kind = 1; s_in_req.n = line[4] ? atoi(line + 4) : -1;
        if (s_in_req.n >= s_ndemo) s_in_req.n = -1;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "test", 4)) {
        s_in_req.kind = 2;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "stats", 5)) {
        mem_report("now");
    } else if (!strncmp(line, "bench", 5)) {
        s_in_req.kind = 5;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
#ifdef ITOFS_ICPROF
    } else if (!strncmp(line, "icprof", 6)) {
        s_in_req.kind = 7;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
#endif
    } else if (!strncmp(line, "first", 5)) {
        s_in_req.kind = 8; s_in_req.n = atoi(line + 5);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "wmode", 5)) {
        s_in_req.kind = 6; s_in_req.n = atoi(line + 5);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (line[0]) {
        printf("commands: say <comma-separated token ids> | style <0-%d> | act <8|16> | demo [0-%d] | test | stats | bench | "
               "wmode <0 direct|1 copy|2 gdma> | first <frames of the first chunk>\n", s_model.n_styles - 1, s_ndemo - 1);
    }
}

static void console_task(void *arg)
{
    (void)arg;
    static char line[4096];
    int len = 0;
    uint8_t ch[64];
    for (;;) {
        int n = uart_read_bytes(UART_NUM_0, ch, sizeof ch, pdMS_TO_TICKS(20));
        for (int i = 0; i < n; i++) {
            if (ch[i] == '\n' || ch[i] == '\r') {
                line[len] = 0;
                if (len) handle_line(line);
                len = 0;
            } else if (len < (int)sizeof line - 1) {
                line[len++] = (char)ch[i];
            }
        }
    }
}

static void button_task(void *arg)
{
    (void)arg;
    gpio_config_t io = { .pin_bit_mask = 1ULL << PIN_BOOT, .mode = GPIO_MODE_INPUT, .pull_up_en = GPIO_PULLUP_ENABLE };
    gpio_config(&io);
    vTaskDelay(pdMS_TO_TICKS(50));
    int prev = gpio_get_level(PIN_BOOT);     // a pin that is already low at boot is not a press
    static req_t r = { .kind = 1, .n = -1 };
    for (;;) {
        int v = gpio_get_level(PIN_BOOT);
        if (prev == 1 && v == 0 && uxQueueMessagesWaiting(s_req_q) == 0) xQueueSend(s_req_q, &r, 0);
        prev = v;
        vTaskDelay(pdMS_TO_TICKS(30));
    }
}

// ------------------------------------------------------------------------------------------------------------------
// boot
// ------------------------------------------------------------------------------------------------------------------
static const esp_partition_t *s_wpart;
static esp_partition_mmap_handle_t s_wmap_h;

// Map the weight blob (only its extent) from the flash partition; returns the mapping and the blob size.
static const void *map_weights(size_t *size)
{
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "weights");
    if (!p) { printf("ERROR no 'weights' partition\n"); return NULL; }
    s_wpart = p;
    uint8_t hdr[16];
    if (esp_partition_read(p, 0, hdr, sizeof hdr) != ESP_OK || memcmp(hdr, "ITF1", 4)) {
        printf("ERROR weights partition is empty or not an ItoFS blob (flash itofs_weights_*.bin at 0x200000)\n"); return NULL;
    }
    uint32_t cfg_len; memcpy(&cfg_len, hdr + 8, 4);
    uint32_t nt;
    if (esp_partition_read(p, 12 + cfg_len, &nt, 4) != ESP_OK || nt > 4096) { printf("ERROR bad blob header\n"); return NULL; }
    size_t ext = 0;
    for (uint32_t i = 0; i < nt; i++) {
        uint8_t e[104];
        if (esp_partition_read(p, 16 + cfg_len + (size_t)i * 104, e, sizeof e) != ESP_OK) return NULL;
        uint32_t off, nb, so, dt, out;
        memcpy(&dt, e + 64, 4); memcpy(&out, e + 76, 4); memcpy(&off, e + 88, 4); memcpy(&nb, e + 92, 4); memcpy(&so, e + 96, 4);
        if (off + nb > ext) ext = off + nb;
        if ((dt == 1 || dt == 3) && so + 4 * out > ext) ext = so + 4 * out;
    }
    if (ext > p->size) { printf("ERROR blob larger than partition\n"); return NULL; }
    *size = ext;
    const void *map;
    const size_t mlen = (ext + 0xFFFF) & ~(size_t)0xFFFF;
    if (esp_partition_mmap(p, 0, mlen, ESP_PARTITION_MMAP_DATA, &map, &s_wmap_h) != ESP_OK) { printf("ERROR mmap weights\n"); return NULL; }
    return map;
}

// Copy the blob to PSRAM if it fits next to the bulk arena (+ margin for the self-test buffer); else keep the mapping.
static const void *place_weights(const void *map, size_t ext, size_t bulk_bytes)
{
    const size_t need = ext + 64 + bulk_bytes + 320 * 1024;
    uint8_t *ps = NULL;
    if (heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) > need)
        ps = heap_caps_aligned_alloc(64, ext + 64, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ps) {
        printf("weights: %u bytes do not fit in PSRAM next to the arenas: running them memory-mapped from flash\n", (unsigned)ext);
        return map;
    }
    int64_t t0 = esp_timer_get_time();
    for (size_t o = 0; o < ext; o += 65536) {     // flash read API in 64 KB pieces
        size_t n = ext - o < 65536 ? ext - o : 65536;
        if (esp_partition_read(s_wpart, o, ps + o, n) != ESP_OK) { printf("ERROR reading weights\n"); heap_caps_free(ps); return map; }
    }
    memset(ps + ext, 0, 64);
    esp_cache_msync(ps, (ext + 64 + 63) & ~(size_t)63, ESP_CACHE_MSYNC_FLAG_DIR_C2M);   // GDMA staging reads PSRAM directly
    esp_partition_munmap(s_wmap_h);
    printf("weights: %u bytes copied flash -> PSRAM in %.0f ms\n", (unsigned)ext, ms(esp_timer_get_time() - t0));
    return ps;
}

static void halt(void) { for (;;) vTaskDelay(1000); }

void app_main(void)
{
    printf("\n==== ItoFS TTS on ESP32-S3 ====\n");
    size_t bs = 0;
    const void *blob = map_weights(&bs);
    if (!blob) halt();
    int e = itofs_model_init(&s_model, blob, bs);
    if (e) { printf("ERROR model: %s\n", itofs_strerror(e)); halt(); }
    {
        itofs_limits_t l0 = { .max_tokens = MAX_TOKENS, .chunk_frames = CHUNK_TARGET / s_model.hop > 0 ? CHUNK_TARGET / s_model.hop : 1 };
        size_t h0, b0;
        itofs_arena_bytes(&s_model, &l0, &h0, &b0);
        const void *placed = place_weights(blob, bs, b0);
        if (placed != blob) {
            blob = placed;
            if ((e = itofs_model_init(&s_model, blob, bs))) { printf("ERROR model: %s\n", itofs_strerror(e)); halt(); }
        }
    }
    s_hop = s_model.hop; s_sr = s_model.sr;
    s_chunk_frames = CHUNK_TARGET / s_hop > 0 ? CHUNK_TARGET / s_hop : 1;
    s_chunk_samples = s_chunk_frames * s_hop;
    s_style = demo_style[0];   // replaced by the first blob demo's style once the demos are loaded
    printf("model: arch %d (%s), %ld int8 + %ld int16 + %ld f32 params, text %d, GRU %d %s, prosody %d, mel head %d x%d -> %d, "
           "decoder %d/%d x%d, n_fft %d, hop %d (%d fps), %d Hz, %d styles%s\n", s_model.arch, s_model.arch == 3 ? "ItoFS v3: mel front + mel vocoder" : "ItoFS v2",
           s_model.n_int8_params, s_model.n_int16_params, s_model.n_f32_params, s_model.text_dim, s_model.rnn_hidden,
           s_model.rnn_bidir ? "bidirectional (whole-sentence text side)" : "forward (incremental text side)", s_model.pros_dim,
           s_model.mel_dim, s_model.mel_layers, s_model.n_mels, s_model.dec_dim, s_model.dec_inter,
           s_model.dec_blocks, s_model.n_fft, s_hop, s_model.fps, s_sr, s_model.n_styles, s_model.style_table ? ", style table" : "");
    {
        const void *d; size_t nb; int dt;
        if (!itofs_blob_find(blob, bs, "selftest", &d, &nb, &dt) && dt == 2 && nb >= 28 && ((const int32_t *)d)[0] == 0x32525453)
            s_st = d;            // 'STR2': n, style, seed, n_samples, fnv32, act_bits, tokens
        if (!itofs_blob_find(blob, bs, "selftest_a16", &d, &nb, &dt) && dt == 2 && nb >= 28 && ((const int32_t *)d)[0] == 0x32525453)
            s_st16 = d;
    }
#ifndef ITOFS_QEMU
    i2s_setup();                     // before the arenas: DMA buffers must get internal RAM first
#else
    printf("QEMU build: I2S disabled (audio is generated and discarded)\n");
#endif
    itofs_limits_t lim = { .max_tokens = MAX_TOKENS, .chunk_frames = s_chunk_frames, .act_bits = ITOFS_ACT_BITS };
    size_t hb, bb;
    itofs_arena_bytes(&s_model, &lim, &hb, &bb);
    // leave room in internal RAM for the task stacks and the PCM / DMA buffers (about 60 KB)
    void *hot = NULL;
    const char *hot_where = "internal SRAM";
    if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) > hb + 64 * 1024)
        hot = heap_caps_aligned_alloc(16, hb, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!hot) { hot = heap_caps_aligned_alloc(16, hb, MALLOC_CAP_SPIRAM); hot_where = "PSRAM (internal SRAM too small)"; }
    void *bulk = heap_caps_aligned_alloc(16, bb, MALLOC_CAP_SPIRAM);
    if (!hot || !bulk || (e = itofs_init(&s_ctx, &s_model, &lim, hot, hb, bulk, bb))) {
        printf("ERROR arena alloc/init (hot %u, bulk %u): %s\n", (unsigned)hb, (unsigned)bb, itofs_strerror(e));
        halt();
    }
    printf("arena: hot %u bytes in %s, bulk %u bytes in PSRAM; chunk %d frames = %d samples; %d-bit activations\n", (unsigned)hb, hot_where,
           (unsigned)bb, s_chunk_frames, s_chunk_samples, s_ctx.act_bits);
    s3_kernels_init();
    s_ctx.qgemm = s3_qgemm;
    s_ctx.par = s3_par;              // row-parallel float work on both cores (bit-identical to one core)
#ifdef ITOFS_ICPROF
    itofs_prof_clock = ic_clock;
#endif
    s_blob = blob; s_blob_size = bs;
    load_demos(blob, bs);
    if (s_ndemo) s_style = s_demo_style[0];
    printf("demos: %d sentences from %s\n", s_ndemo, s_demo_from_blob ? "the weight blob" : "main/demo_sentences.h");

    s_free_q = xQueueCreate(NBUF, sizeof(buf_t));
    s_full_q = xQueueCreate(NBUF + 1, sizeof(buf_t));
    s_done_q = xQueueCreate(1, sizeof(buf_t));
    s_req_q = xQueueCreate(2, sizeof(req_t));
    for (int i = 0; i < NBUF; i++) {
        buf_t b = { heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_INTERNAL), 0 };
        if (!b.pcm) { printf("ERROR no internal RAM for PCM buffers\n"); halt(); }
        xQueueSend(s_free_q, &b, 0);
    }
    uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    mem_report("boot");
    xTaskCreatePinnedToCore(play_task, "itofs_play", 4096, NULL, 12, NULL, 1);
    xTaskCreatePinnedToCore(synth_task, "itofs_synth", 8192, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(console_task, "itofs_con", 4096, NULL, 4, NULL, 0);
    xTaskCreatePinnedToCore(button_task, "itofs_btn", 4096, NULL, 4, NULL, 0);
}
