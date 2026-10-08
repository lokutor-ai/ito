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
#ifndef ITOFS_QEMU
#include "hal/usb_serial_jtag_ll.h"     // the console input of a board connected only through the native USB port (the UART0 pins are unconnected there)
#endif
#include "driver/gpio.h"
#ifndef ITOFS_QEMU
#include "driver/i2s_std.h"
#endif
#include "itofs.h"
#include "itofs_sched.h"
#include "kernels_s3.h"
#include "demo_sentences.h"

#define PIN_BCLK 15
#define PIN_WS 16
#define PIN_DOUT 17
#define PIN_BOOT 0
#define MAX_TOKENS 400
#define I2S_SLICE 480                // samples per I2S write (20 ms = one DMA descriptor); a chunk (up to 300 ms) goes out in slices
#ifndef STEADY_FRAMES
#define STEADY_FRAMES 24             // frames per chunk once the start-up ramp is over (24 frames = 300 ms): every chunk reads each weight
                                     // matrix once from PSRAM, so bigger chunks mean less weight traffic per second of audio
#endif
#define NBUF 6                       // PCM chunk buffers between synthesis and I2S (PSRAM, 14 KB each): with a start delay the player waits while synthesis keeps going
                                     // (the start-delay planner models this pool: production of chunk k waits for the buffer of chunk k - NBUF)
// Start-up schedule (gapless speech from the first chunk). Time to first audio is the time of chunk 0; for playback never to starve, every later chunk
// must be ready before the audio of the chunks before it has played, and each chunk costs a fixed weight pass plus about 6.5 ms (central) per frame.
// A first chunk of 2 frames is out after about 145 ms (central) but its 25 ms of audio is gone long before the 24-frame chunk after it (about 230 ms
// later) is ready, so the speech has a gap. The ramp below starts with 10 frames (125 ms of audio), then 11, 12, 14, 18 and 24: each chunk takes
// less time to make than the one before it takes to play, so playback can start with the first chunk and never starves (estimates, see the
// README). The text side advances in steps of 8 tokens meanwhile (a 24-token step would make one chunk 65 ms slower than its neighbours). The audio
// is bit-identical to any other chunking (host test G).
#ifndef RAMP_LIST
#define RAMP_LIST 11, 12, 14, 18     // frames of chunks 1, 2, ... after the first one, then STEADY_FRAMES
#endif
#ifndef TEXT_STEP_LIST
#define TEXT_STEP_LIST 8, 8, 8, 8, 8, 8, 8, 8   // text-side tokens added per step in chunk 0, 1, ... (0 or past the list: the engine's default, 24)
#endif
#ifndef FIRST_FRAMES
#define FIRST_FRAMES 10              // frames of the first chunk (10 = 125 ms of audio). `first <n>` changes it (n >= chunk = no ramp at all); `first 2` is the
                                     // lowest-latency start (about 145 ms central) but not gapless
#endif
#ifndef START_DELAY_MS
#define START_DELAY_MS -1            // -1: the start delay is planned per utterance from the boot calibration (itofs_sched.h); >= 0 fixes it in ms (the `delay <ms>` command does the same at run time)
#endif
#ifndef ITOFS_ACT_BITS
#define ITOFS_ACT_BITS 8             // activations of the int8-weight layers: 8 (one int8 plane, default) or 16; `act` command
#endif

static itofs_model_t s_model;
static itofs_ctx_t s_ctx;
static int s_hop, s_sr, s_chunk_frames, s_chunk_samples, s_style, s_first = FIRST_FRAMES;
static const int s_ramp_default[] = { RAMP_LIST };
static const int *s_ramp = s_ramp_default;
static int s_nramp = (int)(sizeof s_ramp_default / sizeof s_ramp_default[0]);
static const int s_tstep[] = { TEXT_STEP_LIST };

// frames to request for chunk k (0, 1, ...) of an utterance: `first` frames, then the ramp, then the steady-state chunk.
// first <= 0 or >= a full chunk turns the ramp off.
static int chunk_frames_for(int k)
{
    if (s_first <= 0 || s_first >= s_chunk_frames) return s_chunk_frames;
    if (k == 0) return s_first;
    if (k - 1 < s_nramp) return s_ramp[k - 1] < s_chunk_frames ? s_ramp[k - 1] : s_chunk_frames;
    return s_chunk_frames;
}
// text-side tokens per step while chunk k is made: small steps early (a step is a whole pass over the text weights, plus the tokens),
// so the start-up chunks stay even; the engine's default (24 tokens) once the ramp is over. Audio is identical for every value.
static int text_step_for(int k)
{
    if (s_first <= 0 || s_first >= s_chunk_frames) return 0;
    return k < (int)(sizeof s_tstep / sizeof s_tstep[0]) ? s_tstep[k] : 0;
}
static const int32_t *s_st, *s_st16; // self-test records from the blob ("selftest": default 8-bit activations,
                                     // "selftest_a16": 16-bit); NULL if absent

typedef struct { int16_t *pcm; int n; } buf_t;      // n = samples, 0 = end of utterance
static QueueHandle_t s_free_q, s_full_q, s_req_q, s_done_q;
static volatile int s_underruns, s_playing;
static volatile int64_t s_utt_t0;       // esp_timer time (us) at which the current utterance was handed to the engine
static int s_delay_override_ms = START_DELAY_MS;   // >= 0: hold the DAC until this long after the text arrived, whatever the plan says (`delay <ms>`); -1: use the plan (`delay auto`)
static volatile int64_t s_cur_delay_us;            // the planned start delay of the current utterance (from its text arrival); raised while the player waits if the chunks come slower than calibrated
static int s_active = -1, s_selected = -1, s_degraded;   // weight set in use / chosen by the boot calibration / degraded flag (see the weight-set section)
#define PLAN_MAX_DELAY_US 4000000.0
static double s_obs_us[ITOFS_SCHED_MAXCH];         // production times of the chunks of the current utterance so far

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
    int16_t *st = heap_caps_malloc(sizeof(int16_t) * 2 * (size_t)I2S_SLICE + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);   // one 20 ms slice
    buf_t b;
    int64_t end_us = 0;              // when the audio handed to the I2S so far has been played out (the DMA ring holds ~120 ms, and i2s_channel_write returns once the data is queued, not played)
    for (;;) {
        int starved = 0;
        if (s_playing && xQueueReceive(s_full_q, &b, 0) != pdTRUE) {
            starved = 1;             // the queue is empty: an underrun only if the DAC has also run out of audio (checked below; found on the board: the ring made every late chunk look like a gap)
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        } else if (!s_playing) {
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        }
        if (b.n == 0) { s_playing = 0; xQueueSend(s_done_q, &b, portMAX_DELAY); continue; }
        if (!s_playing) {                                  // pre-roll: the first chunk waits for the delay, later chunks keep arriving meanwhile
            for (;;) {                                     // (re-read in <= 10 ms steps: the synthesis task may raise the planned delay)
                const int64_t d = s_delay_override_ms >= 0 ? (int64_t)s_delay_override_ms * 1000 : s_cur_delay_us;
                const int64_t wait = s_utt_t0 + d - esp_timer_get_time();
                if (wait <= 1000) break;
                vTaskDelay(pdMS_TO_TICKS((wait > 10000 ? 10000 : wait + 999) / 1000));
            }
        }
        {
            const int64_t now = esp_timer_get_time();
            if (!s_playing) end_us = now;
            else if (starved && now > end_us + 2000) { s_underruns++; end_us = now; }      // synthesis did not keep up: the DAC played silence meanwhile
        }
        s_playing = 1;
        end_us += (int64_t)b.n * 1000000 / s_sr;
#ifndef ITOFS_QEMU
        for (int o = 0; o < b.n; o += I2S_SLICE) {
            const int m = b.n - o < I2S_SLICE ? b.n - o : I2S_SLICE;
            for (int i = 0; i < m; i++) { st[2 * i] = b.pcm[o + i]; st[2 * i + 1] = b.pcm[o + i]; }
            size_t wr = 0;
            i2s_channel_write(s_tx, st, sizeof(int16_t) * 2 * (size_t)m, &wr, portMAX_DELAY);
        }
#else
        (void)st;
#endif
        xQueueSend(s_free_q, &b, portMAX_DELAY);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// synthesis
// ------------------------------------------------------------------------------------------------------------------
typedef struct { double ttfa_ms, begin_ms, compute_ms, audio_s; int frames, tok_first; double plan_delay_ms; int plan_underruns, plan_feasible; } stats_t;

// Synthesise tokens; stream chunks to the player (play=1) or into out[] (play=0).
static int tier_plan(int n_tok, int nobs, itofs_plan_t *pl);       // below: start-delay plan from the active weight set's calibration
static int speak(const int *tok, int n, int style, uint32_t seed, int play, int16_t *out, long out_max, stats_t *S)
{
    S->tok_first = 0; S->plan_delay_ms = -1; S->plan_underruns = 0; S->plan_feasible = 1;
    s_cur_delay_us = 0;
    int64_t t0 = esp_timer_get_time();
    s_utt_t0 = t0;
    s3_stats_reset();
    int T = itofs_begin(&s_ctx, tok, n, style, seed);
    int64_t t1 = esp_timer_get_time();
    if (T < 0) { printf("ERROR itofs_begin: %s\n", itofs_strerror(T)); return T; }
    int64_t compute = t1 - t0, ttfa = -1;
    long pos = 0;
    int kch = 0;
    s_underruns = 0;
    for (;;) {
        buf_t b = { NULL, 0 };
        if (play) xQueueReceive(s_free_q, &b, portMAX_DELAY);
        int16_t *dst = play ? b.pcm : out + pos;
        s_ctx.text_step = text_step_for(kch);
        const int want = chunk_frames_for(kch++) * s_hop;                                  // low-latency first chunk, ramp, steady state
        int room = play ? want : (int)((out_max - pos) < want ? (out_max - pos) : want);
        int64_t a = esp_timer_get_time();
        int got = room >= s_hop ? itofs_next_chunk(&s_ctx, dst, room) : 0;
        int64_t e = esp_timer_get_time();
        compute += e - a;
        if (got <= 0) { if (play) xQueueSend(s_free_q, &b, portMAX_DELAY); break; }
        if (ttfa < 0) { ttfa = e - t0; S->tok_first = s_ctx.n_done; }
        pos += got;
        if (play) {
            if (kch - 1 < ITOFS_SCHED_MAXCH) s_obs_us[kch - 1] = (double)(e - a);
            if (!s_playing) {          // adaptive pre-buffer: plan (kch == 1) or re-plan the start delay with the chunk times actually seen in this utterance
                itofs_plan_t pl;
                if (tier_plan(n, kch < ITOFS_SCHED_MAXCH ? kch : ITOFS_SCHED_MAXCH, &pl)) {
                    s_cur_delay_us = (int64_t)pl.start_delay_us;
                    S->plan_delay_ms = pl.start_delay_us / 1e3; S->plan_underruns = pl.underruns; S->plan_feasible = pl.feasible;
                }
            }
            b.n = got; xQueueSend(s_full_q, &b, portMAX_DELAY);
        }
    }
    if (play) {
        buf_t end = { NULL, 0 };
        xQueueSend(s_full_q, &end, portMAX_DELAY);
        xQueueReceive(s_done_q, &end, portMAX_DELAY);
    }
    s_ctx.text_step = 0;
    S->begin_ms = ms(t1 - t0); S->ttfa_ms = ms(ttfa); S->compute_ms = ms(compute);
    S->frames = s_ctx.T; S->audio_s = (double)pos / s_sr; (void)T;
    return (int)pos;
}

static void print_timing(const char *what, const stats_t *S, int n_tok, int style)
{
    printf("TIMING %s: %d tokens, style %d, %.2f s audio | first chunk %.1f ms (begin %.1f ms; %s text side, %d tokens final at first chunk) | "
           "compute %.1f ms, RTF %.3f | underruns %d (start delay %s %.0f ms%s) | weight set %d | weight staging %s: %ld GEMM calls, %ld announced ahead, %ld not\n",
           what, n_tok, style, S->audio_s, S->ttfa_ms, S->begin_ms, s_ctx.text_incr ? "incremental" : "whole-sentence", S->tok_first,
           S->compute_ms, S->compute_ms / 1000.0 / S->audio_s, s_underruns, s_delay_override_ms >= 0 ? "fixed" : "planned",
           s_delay_override_ms >= 0 ? (double)s_delay_override_ms : S->plan_delay_ms, s_delay_override_ms >= 0 || S->plan_feasible ? "" : ", model predicts underruns", s_active, s3_wmode_name(s3_wmode),
           s3_stats.calls, s3_stats.pf_hit, s3_stats.pf_miss + s3_stats.pf_free);
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
static int s_tested, s_selftest_ok = 1;     // the active weight set has passed its self-test / result of the last self-test
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
        if (!ok) s_selftest_ok = 0;
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
    s_selftest_ok = 1;
    if (!s_st) { printf("SELFTEST skipped: this weight blob has no self-test record\n"); s_tested = 1; return; }
    self_test_one(s_st, both, 1);
    if (both && s_st16) self_test_one(s_st16, 0, 0);
    s_tested = 1;
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

// weight staging mode; the engine announces the next GEMM call's weights only when the kernel can use it (mode 3)
static int set_wmode(int m)
{
    const int r = s3_set_wmode(m);
    if (!r) s_ctx.qnext = m == 3 ? s3_prefetch : NULL;
    return r;
}

static void tts_bench(void)
{
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_SPIRAM);   // one chunk, overwritten
    if (!pcm) { printf("BENCH ERROR: no memory for the PCM buffer\n"); return; }
    double best_rtf = 1e30, sum_ttfa[4] = {0}, sum_rtf[4] = {0}, gops[4] = {0};
    int best = s3_wmode, nok[4] = {0};
    double tt[4][MAX_DEMOS] = {{0}}, tok_n[MAX_DEMOS] = {0};
    const int saved = s3_wmode;
    uint32_t href[MAX_DEMOS] = {0};
    for (int mode = 0; mode < 4; mode++) {
        int mismatch = 0;
        if (set_wmode(mode)) { printf("BENCH_TTS wmode %s: unavailable\n", s3_wmode_name(mode)); continue; }
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
            int kc = 0;
            s_ctx.text_step = text_step_for(kc);
            int got = itofs_next_chunk(&s_ctx, pcm, chunk_frames_for(kc++) * s_hop);
            uint32_t hh = 2166136261u;
            for (int i = 0; i < got; i++) { hh ^= (uint16_t)pcm[i]; hh *= 16777619u; }
            const int64_t t2 = esp_timer_get_time();
            const double o8f = ctx_ops(0) - o8a, off = ctx_ops(1) - ofa, gus_first = (double)s3_stats.us;
            while (got > 0) {
                pos += got; s_ctx.text_step = text_step_for(kc); got = itofs_next_chunk(&s_ctx, pcm, chunk_frames_for(kc++) * s_hop);
                for (int i = 0; i < got; i++) { hh ^= (uint16_t)pcm[i]; hh *= 16777619u; }
            }
            const int64_t t3 = esp_timer_get_time();
            // every staging mode must give the same audio as direct reads; a mode that does not is excluded from the choice
            if (mode == 0) href[k] = hh;
            else if (hh != href[k]) { printf("BENCH_TTS wmode %s demo %d: OUTPUT MISMATCH (hash %08lx, direct %08lx): mode excluded\n", s3_wmode_name(mode), k, (unsigned long)hh, (unsigned long)href[k]); mismatch = 1; }
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
        if (mismatch) nok[mode] = 0;
        if (nok[mode] && sum_rtf[mode] / nok[mode] < best_rtf) { best_rtf = sum_rtf[mode] / nok[mode]; best = mode; }
        if (nok[mode] >= 2) {     // TTFA = a + b * tokens (least squares over the demos)
            double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = nok[mode];
            for (int k = 0; k < m; k++) { sx += tok_n[k]; sy += tt[mode][k]; sxx += tok_n[k] * tok_n[k]; sxy += tok_n[k] * tt[mode][k]; }
            const double b = (m * sxy - sx * sy) / (m * sxx - sx * sx + 1e-9), a = (sy - b * sx) / m;
            printf("BENCH_TTFA_MODEL wmode %s: first chunk = %.1f ms + %.3f ms/token -> under 200 ms up to %d tokens\n",
                   s3_wmode_name(mode), a, b, b > 0 ? (int)((200 - a) / b) : -1);
        }
    }
    set_wmode(nok[best] ? best : saved);
    printf("BOARD_SUMMARY fw=v3bench arch=%d act=%d best_wmode=%s", s_model.arch, s_ctx.act_bits, s3_wmode_name(s3_wmode));
    for (int mode = 0; mode < 4; mode++)
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
                                         "gtENC", "gtGRU", "gtPROJ", "gtDUR", "gtDOUT",
                                         "pBLKDW", "pBLKGELU", "pBLKRES", "pRESCALE", "pBIAS", "pGATHPUT", "pLN", "pHEADB", "pHFTB", "pSRCB", "pOLAB", "pCLNB", "pRESC1"};
static void ic_prof_print(const char *what, const double *a, const double *b)
{
    printf("ICPROF_STAGES %s", what);
    for (int k = 0; k < ITOFS_PROF_N; k++) printf(" %s %.0f", ic_pn[k], b[k] - (a ? a[k] : 0));
    printf("\n");
}

static int ic_fr(const int *sch, int k) { for (int i = 0; i <= k; i++) if (!sch[i]) return s_chunk_frames; return sch[k]; }
#define IC_MAXCH 160
static float s_icch[IC_MAXCH][4];    // per chunk: samples, critical-path ticks (mode 1), weight bytes, f32 MACs
// sch: frames of chunk 0, 1, ... (0 ends the list: the steady-state chunk follows); sid is its id in the log
static int s_ic_all;     // icprof 2: trace every chunk of the self-test sentence and the demos (start-up schedule comparison)
static int ic_ts(const int *ts, int k) { for (int i = 0; i <= k; i++) if (!ts[i]) return 0; return ts[k]; }   // text-side tokens per step of chunk k (0 = engine default)
static void ic_one(const char *tag, const int *tok, int n, int style, uint32_t seed, int mode, int16_t *pcm, const int *sch, const int *tsch, int sid)
{
    itofs_prof_clock = ic_clock;
    memset(itofs_prof, 0, sizeof itofs_prof);
    double pf[ITOFS_PROF_N];
    s3_dual_enabled = mode != 0; s3_ic_serial = mode == 1;
    memset(&s3_ic, 0, sizeof s3_ic); s3_stats_reset();
    icsnap_t z, f, e;
    ic_snap(&z, 0);
    vTaskDelay(1);
    const int do_trace = mode == 1 && (s_ic_all ? 1 : (sid == 0 && !strcmp(tag, "demo1")));
    if (do_trace) {
        if (!s3_trace) { s3_trace_max = 16000; s3_trace = heap_caps_malloc(sizeof(s3_ev_t) * s3_trace_max, MALLOC_CAP_SPIRAM); }
        s3_trace_reset(); s3_trace_on = s3_trace != NULL;
    }
    const uint32_t c0 = esp_cpu_get_cycle_count();
    int T = itofs_begin(&s_ctx, tok, n, style, seed);
    if (T < 0) { printf("ICPROF ERROR %s\n", itofs_strerror(T)); return; }
    int kc = 0;
    s_ctx.text_step = ic_ts(tsch, kc);
    int got = itofs_next_chunk(&s_ctx, pcm, ic_fr(sch, kc++) * s_hop);
    if (do_trace) s3_trace_mark();
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
        s_ctx.text_step = ic_ts(tsch, kc);
        got = itofs_next_chunk(&s_ctx, pcm, ic_fr(sch, kc++) * s_hop);
        if (do_trace) s3_trace_mark();
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
    if (do_trace) {
        s3_trace_on = 0;
        int ch = 0;
        for (int i = 0; i < s3_trace_n && (s_ic_all || ch < 6); i++) {
            const s3_ev_t *v = &s3_trace[i];
            printf("ICTRACE %s %d %d %d %u %u %u %u %u %d %d %d\n", tag, sid, ch, v->kind, (unsigned)v->gap, (unsigned)v->h0, (unsigned)v->h1, (unsigned)v->ovh, (unsigned)v->bytes, v->rows, v->in, v->out);
            if (v->kind == 9) ch++;
        }
        printf("ICTRACE_END %s %d events %d\n", tag, sid, s3_trace_n);
    }
    printf("ICPROF %s mode %s first %d sched %d tokens %d audio %.4f chunks %d fnv %08lx", tag, mode == 0 ? "1core" : mode == 1 ? "2core-serial" : "2core-par",
           sch[0] ? sch[0] : s_chunk_frames, sid, n, (double)pos / s_sr, nch, (unsigned long)h);
    ic_print("first", &z, &f);
    ic_print("rest", &f, &e);
    printf("\n");
    if (mode == 0) { ic_prof_print("first", NULL, pf); ic_prof_print("rest", pf, itofs_prof); }
    if (mode == 1) {                 // per chunk: samples / critical-path ticks / weight bytes / f32 MACs
        printf("ICPROF_CHUNKS %s sched %d:", tag, sid);
        for (int k = 0; k < nch && k < IC_MAXCH; k++) printf(" %.0f/%.0f/%.0f/%.0f", s_icch[k][0], s_icch[k][1], s_icch[k][2], s_icch[k][3]);
        printf("\n");
    }
}

static void icprof(int quick)
{
    ic_calib();
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_SPIRAM);
    if (!pcm) { printf("ICPROF ERROR no memory\n"); return; }
    printf("ICPROF begin: chunk %d frames, %d-bit activations, wmode %s\n", s_chunk_frames, s_ctx.act_bits, s3_wmode_name(s3_wmode));
    // (mode, schedule): 1 core; 2 cores serialised (exact per-core split) with several start-up ramps; 2 cores in parallel
    static const int sched[][12] = { {2, 0}, {2, 8, 16, 0}, {2, 8, 0}, {2, 12, 0}, {2, 6, 12, 0}, {1, 0}, {8, 0} };
    // start-up schedules compared by `icprof 2` (every chunk traced, all four sentences): frames per chunk (then 24) and text-side tokens per step
    // (then the engine default). 0 = the old 2-frame first chunk (fastest first sound, but a gap follows), 1 = shipped, 2 / 3 = a longer / shorter first chunk
    static const int sched2[][12] = { {2, 0}, {10, 11, 12, 14, 18, 0}, {12, 14, 17, 21, 0}, {8, 9, 10, 12, 15, 19, 0} };
    static const int tsched2[][12] = { {0}, {8, 8, 8, 8, 8, 8, 8, 8, 0}, {8, 8, 8, 8, 8, 8, 8, 8, 0}, {8, 8, 8, 8, 8, 8, 8, 8, 0} };
    static const int zts[12] = {0};
    static const int runs[][2] = { {0, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 1} };
    if (quick >= 2) {
        s_ic_all = 1;
        for (int sid = 0; sid < (int)(sizeof sched2 / sizeof sched2[0]); sid++) {
            if (s_st) {
                int n = s_st[1];
                for (int i = 0; i < n; i++) s_tok_buf[i] = s_st[7 + i];
                ic_one("selftest", s_tok_buf, n, s_st[2], (uint32_t)s_st[3], 1, pcm, sched2[sid], tsched2[sid], sid);
            }
            for (int k = 0; k < s_ndemo; k++) {
                char tg[16]; snprintf(tg, sizeof tg, "demo%d", k);
                demo_tokens(k, s_tok_buf);
                ic_one(tg, s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, 1, pcm, sched2[sid], tsched2[sid], sid);
            }
        }
        s_ic_all = 0; s3_dual_enabled = 1; s3_ic_serial = 0;
        heap_caps_free(pcm);
        printf("ICPROF end\n");
        return;
    }
    for (int ri = 0; ri < (int)(sizeof runs / sizeof runs[0]); ri++) {
        const int mode = runs[ri][0], sid = runs[ri][1];
        if (quick && ri > 1) break;
        if (s_st) {
            int n = s_st[1];
            for (int i = 0; i < n; i++) s_tok_buf[i] = s_st[7 + i];
            ic_one("selftest", s_tok_buf, n, s_st[2], (uint32_t)s_st[3], mode, pcm, sched[sid], zts, sid);
        }
        for (int k = 0; k < s_ndemo; k++) {
            char tg[16]; snprintf(tg, sizeof tg, "demo%d", k);
            demo_tokens(k, s_tok_buf);
            ic_one(tg, s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, mode, pcm, sched[sid], zts, sid);
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

// ------------------------------------------------------------------------------------------------------------------
// weight sets ("tiers") and the boot self-calibration
//
// Up to three ItoFS blobs sit in flash, best quality first: `weights` (main, int8), `weights_b` (main, int4 blocks), `weights_c` (light: 4 blocks, int4).
// Only one is in PSRAM at a time. At boot the firmware activates them in that order, synthesises a representative sentence (the blob's self-test sentence)
// without playback with the shipped chunk schedule, records the real time of every chunk (compute plus weight fetch from PSRAM), turns that into a measured
// real-time factor, and keeps the first set whose RTF is <= 0.85 (itofs_sched.h). If none is, it keeps the fastest and says so; if that one is still
// >= 0.95 it prints a WARNING line and sets the degraded flag instead of silently stuttering. The same measured chunk times plan, per utterance, the
// playback start delay (an adaptive jitter pre-buffer: the first audio is released as soon as the model says the speaker cannot run dry).
// `simtime a b c` replaces the measured chunk times by modelled ones (RTF a, b, c for the three sets) so the whole logic can be exercised in QEMU.
// ------------------------------------------------------------------------------------------------------------------
#define MAX_TIERS 3
typedef struct {
    const esp_partition_t *p; size_t ext; int valid;
    int calibrated; double rtf, rtf_steady, rtf_whole; itofs_cal_t cal;
    int blocks; long n_int4; char desc[48];
} tier_t;
static tier_t s_tier[MAX_TIERS];
static const char *const s_tier_label[MAX_TIERS] = { "weights", "weights_b", "weights_c" };
static itofs_tier_status_t s_tstatus = ITOFS_TIER_OK;
static uint8_t *s_wbuf;                       // PSRAM copy of the active set (NULL: running memory-mapped from flash)
static esp_partition_mmap_handle_t s_map_h; static int s_mapped;
static void *s_hot, *s_bulk; static size_t s_hb, s_bb;
static itofs_limits_t s_lim;
static int s_sim; static double s_sim_rtf[MAX_TIERS] = { 0.9, 0.8, 0.7 };

static int tier_ext(const esp_partition_t *p, size_t *ext, int quiet)
{
    uint8_t hdr[16];
    if (esp_partition_read(p, 0, hdr, sizeof hdr) != ESP_OK || memcmp(hdr, "ITF1", 4)) {
        if (!quiet) printf("ERROR weights partition '%s' is empty or not an ItoFS blob (flash the itofs_weights_*.bin at its offset)\n", p->label);
        return -1;
    }
    uint32_t cfg_len; memcpy(&cfg_len, hdr + 8, 4);
    uint32_t nt;
    if (esp_partition_read(p, 12 + cfg_len, &nt, 4) != ESP_OK || nt > 4096) { printf("ERROR bad blob header in '%s'\n", p->label); return -1; }
    size_t e = 0;
    for (uint32_t i = 0; i < nt; i++) {
        uint8_t ent[104];
        if (esp_partition_read(p, 16 + cfg_len + (size_t)i * 104, ent, sizeof ent) != ESP_OK) return -1;
        uint32_t off, nb, so, dt, out;
        memcpy(&dt, ent + 64, 4); memcpy(&out, ent + 76, 4); memcpy(&off, ent + 88, 4); memcpy(&nb, ent + 92, 4); memcpy(&so, ent + 96, 4);
        if (off + nb > e) e = off + nb;
        if ((dt == 1 || dt == 3 || dt == 4) && so + 4 * out > e) e = so + 4 * out;
    }
    if (e > p->size) { printf("ERROR blob in '%s' larger than its partition\n", p->label); return -1; }
    *ext = e;
    return 0;
}

// dimensions of a set from the flash mapping (no copy): arena sizes, hop, sample rate
static int tier_dims(int i, size_t *hb, size_t *bb, int *hop, int *sr)
{
    const tier_t *T = &s_tier[i];
    esp_partition_mmap_handle_t h; const void *map;
    if (esp_partition_mmap(T->p, 0, (T->ext + 0xFFFF) & ~(size_t)0xFFFF, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) { printf("ERROR mmap '%s'\n", T->p->label); return -1; }
    itofs_model_t *m = &s_model;                          // (the global one, used as scratch before the first activation: the struct is far larger than the main task's stack margin)
    int e = itofs_model_init(m, map, T->ext);
    if (!e) { itofs_arena_bytes(m, &s_lim, hb, bb); *hop = m->hop; *sr = m->sr; }
    else printf("ERROR model in '%s': %s\n", T->p->label, itofs_strerror(e));
    esp_partition_munmap(h);
    return e ? -1 : 0;
}

// make set i the active one: weights into PSRAM (or the flash mapping), model parsed, engine context initialised over the shared arenas
static int activate_tier(int i)
{
    tier_t *T = &s_tier[i];
    s3_stream_reset();                                    // weight staging idle before the weights under it change
    const void *blob;
    int64_t t0 = esp_timer_get_time();
    if (s_mapped) { esp_partition_munmap(s_map_h); s_mapped = 0; }
    if (s_wbuf) {
        for (size_t o = 0; o < T->ext; o += 65536) {      // flash read API in 64 KB pieces
            size_t n = T->ext - o < 65536 ? T->ext - o : 65536;
            if (esp_partition_read(T->p, o, s_wbuf + o, n) != ESP_OK) { printf("ERROR reading weights '%s'\n", T->p->label); return -1; }
        }
        memset(s_wbuf + T->ext, 0, 64);
        esp_cache_msync(s_wbuf, (T->ext + 64 + 63) & ~(size_t)63, ESP_CACHE_MSYNC_FLAG_DIR_C2M);   // GDMA staging reads PSRAM directly
        blob = s_wbuf;
        printf("weights: set %d ('%s'): %u bytes copied flash -> PSRAM in %.0f ms\n", i, T->p->label, (unsigned)T->ext, ms(esp_timer_get_time() - t0));
    } else {
        const void *map;
        if (esp_partition_mmap(T->p, 0, (T->ext + 0xFFFF) & ~(size_t)0xFFFF, ESP_PARTITION_MMAP_DATA, &map, &s_map_h) != ESP_OK) { printf("ERROR mmap weights\n"); return -1; }
        s_mapped = 1; blob = map;
        printf("weights: set %d ('%s'): %u bytes run memory-mapped from flash (PSRAM too small for a copy)\n", i, T->p->label, (unsigned)T->ext);
    }
    int e = itofs_model_init(&s_model, blob, T->ext);
    if (e) { printf("ERROR model: %s\n", itofs_strerror(e)); return -1; }
    if ((e = itofs_init(&s_ctx, &s_model, &s_lim, s_hot, s_hb, s_bulk, s_bb))) { printf("ERROR engine init: %s\n", itofs_strerror(e)); return -1; }
    s_ctx.qgemm = s3_qgemm;
    s_ctx.qgemm4 = s3_qgemm4;
    s_ctx.par = s3_par;                                   // row-parallel float work on both cores (bit-identical to one core)
    s_ctx.qnext = s3_wmode == 3 ? s3_prefetch : NULL;
#ifdef ITOFS_ICPROF
    itofs_prof_clock = ic_clock;
#endif
    s_blob = blob; s_blob_size = T->ext;
    T->blocks = s_model.dec_blocks; T->n_int4 = s_model.n_int4_params;
    snprintf(T->desc, sizeof T->desc, "vocoder %d/%d x%d, %s", s_model.dec_dim, s_model.dec_inter, s_model.dec_blocks, s_model.n_int4_params ? "int4 blocks" : "int8");
    printf("model: arch %d (%s), %ld int8 + %ld int4 + %ld int16 + %ld f32 params, text %d, GRU %d %s, prosody %d, mel head %d x%d -> %d, "
           "decoder %d/%d x%d, n_fft %d, hop %d (%d fps), %d Hz, %d styles%s, blob format %d\n", s_model.arch, s_model.arch == 3 ? "ItoFS v3: mel front + mel vocoder" : "ItoFS v2",
           s_model.n_int8_params, s_model.n_int4_params, s_model.n_int16_params, s_model.n_f32_params, s_model.text_dim, s_model.rnn_hidden,
           s_model.rnn_bidir ? "bidirectional (whole-sentence text side)" : "forward (incremental text side)", s_model.pros_dim,
           s_model.mel_dim, s_model.mel_layers, s_model.n_mels, s_model.dec_dim, s_model.dec_inter,
           s_model.dec_blocks, s_model.n_fft, s_model.hop, s_model.fps, s_model.sr, s_model.n_styles, s_model.style_table ? ", style table" : "", s_model.format);
    s_st = s_st16 = NULL;
    {
        const void *d; size_t nb; int dt;
        if (!itofs_blob_find(blob, T->ext, "selftest", &d, &nb, &dt) && dt == 2 && nb >= 28 && ((const int32_t *)d)[0] == 0x32525453) s_st = d;        // 'STR2'
        if (!itofs_blob_find(blob, T->ext, "selftest_a16", &d, &nb, &dt) && dt == 2 && nb >= 28 && ((const int32_t *)d)[0] == 0x32525453) s_st16 = d;
    }
    s_ndemo = 0; s_demo_from_blob = 0;
    load_demos(blob, T->ext);
    if (s_ndemo) s_style = s_demo_style[0];
    s_active = i; s_tested = 0;
    return 0;
}

// real chunk times of the active set: the self-test sentence, no playback, the shipped schedule
static void calibrate_tier(int i)
{
    tier_t *T = &s_tier[i];
    itofs_cal_t *c = &T->cal;
    memset(c, 0, sizeof *c);
    T->calibrated = 0; T->rtf = -1;
    int n = 0, style = s_style; uint32_t seed = 1;
    if (s_st) { n = s_st[1]; style = s_st[2]; seed = (uint32_t)s_st[3]; for (int k = 0; k < n; k++) s_tok_buf[k] = s_st[7 + k]; }
    else if (s_ndemo) { n = s_demo_len[0]; style = s_demo_style[0]; demo_tokens(0, s_tok_buf); }
    if (n <= 0) { printf("CALIB set %d: no representative sentence in this blob\n", i); return; }
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_SPIRAM);
    if (!pcm) { printf("CALIB ERROR: no memory\n"); return; }
    c->hop = s_hop; c->sr = s_sr; c->steady_frames = s_chunk_frames; c->n_tok = n;
    s3_stats_reset();
    const int act_saved = s_ctx.act_bits;
    if (s_st) s_ctx.act_bits = s_st[6];
    int64_t t0 = esp_timer_get_time();
    const int Tfr = itofs_begin(&s_ctx, s_tok_buf, n, style, seed);
    int64_t t1 = esp_timer_get_time();
    if (Tfr < 0) { printf("CALIB ERROR itofs_begin: %s\n", itofs_strerror(Tfr)); s_ctx.act_bits = act_saved; heap_caps_free(pcm); return; }
    c->begin_us = (double)(t1 - t0);
    int kc = 0;
    for (;;) {
        s_ctx.text_step = text_step_for(kc);
        const int want = chunk_frames_for(kc) * s_hop;
        const int64_t a = esp_timer_get_time();
        const int got = itofs_next_chunk(&s_ctx, pcm, want);
        const int64_t e = esp_timer_get_time();
        if (got <= 0) break;
        if (kc < ITOFS_SCHED_MAXCH) { c->frames[kc] = got / s_hop; c->t_us[kc] = (double)(e - a); kc++; } else break;
    }
    s_ctx.text_step = 0; s_ctx.act_bits = act_saved;
    c->n = kc;
    if (s_sim) {          // timing injection: modelled chunk times for an RTF of s_sim_rtf[i] (a fixed share of 35 % per chunk, the rest per frame)
        const double t24 = s_sim_rtf[i] * c->steady_frames * s_hop / (double)s_sr * 1e6;
        for (int k = 0; k < kc; k++) c->t_us[k] = t24 * (0.35 + 0.65 * c->frames[k] / (double)c->steady_frames);
        c->begin_us = 6000.0;
    }
    double steady = 0, whole = 0;
    T->rtf = itofs_cal_rtf(c, &steady, &whole);
    T->rtf_steady = steady > 0 ? steady / (c->steady_frames * (double)s_hop / s_sr * 1e6) : -1; T->rtf_whole = whole;
    T->calibrated = T->rtf >= 0;
    printf("CALIB set %d ('%s', %s)%s: %d tokens, %d chunks | first chunk %.1f ms (begin %.1f ms) | steady chunk %.1f ms of %.0f ms audio: RTF steady %.3f, whole %.3f -> measured RTF %.3f | weight staging %s\n",
           i, T->p->label, T->desc, s_sim ? " [SIMULATED TIMES]" : "", n, kc, kc ? c->t_us[0] / 1e3 : 0.0, c->begin_us / 1e3, steady / 1e3,
           c->steady_frames * (double)s_hop / s_sr * 1e3, T->rtf_steady, whole, T->rtf, s3_wmode_name(s3_wmode));
    heap_caps_free(pcm);
}

static int tier_plan(int n_tok, int nobs, itofs_plan_t *pl)
{
    if (s_delay_override_ms >= 0 || s_active < 0 || !s_tier[s_active].calibrated || s_tier[s_active].cal.n < 2) return 0;   // fixed delay, or nothing measured: no plan
    itofs_sched_plan(&s_tier[s_active].cal, n_tok, 0, NBUF, ITOFS_SCHED_SAFETY, PLAN_MAX_DELAY_US, s_obs_us, nobs, pl);
    return 1;
}

static void self_test(int both);
static void tiers_select(void)
{
    double rtf[MAX_TIERS];
    itofs_tier_status_t st = ITOFS_TIER_OK;
    int pick = -1;
    for (int attempt = 0; attempt < MAX_TIERS; attempt++) {
        for (int i = 0; i < MAX_TIERS; i++) rtf[i] = -1;
        for (int i = 0; i < MAX_TIERS; i++) {                 // best quality first; stop at the first set that meets the target
            if (!s_tier[i].valid) continue;
            if (s_active != i && activate_tier(i)) { s_tier[i].valid = 0; continue; }
            calibrate_tier(i);
            rtf[i] = s_tier[i].rtf;
            if (rtf[i] >= 0 && rtf[i] <= ITOFS_RTF_TARGET) break;
        }
        pick = itofs_sched_pick(rtf, MAX_TIERS, ITOFS_RTF_TARGET, ITOFS_RTF_DEGRADED, &st);
        if (pick < 0) { printf("TIER_SELECT ERROR: no weight set could be measured\n"); s_degraded = 1; return; }
        if (pick != s_active && activate_tier(pick)) { s_tier[pick].valid = 0; continue; }
        if (!s_tested) {                                      // the chosen set must reproduce the host engine's PCM bit for bit
            self_test(0);
            if (!s_selftest_ok) { printf("TIER_SELECT: weight set %d FAILED its self-test and is excluded\n", pick); s_tier[pick].valid = 0; s_tested = 0; continue; }
        }
        break;
    }
    s_selected = pick; s_tstatus = st; s_degraded = st == ITOFS_TIER_DEGRADED;
    printf("TIER_SELECT chosen=%d ('%s', %s) measured_rtf=%.3f target=%.2f status=%s degraded=%d%s | measured:", pick, s_tier[pick].p->label, s_tier[pick].desc, rtf[pick],
           ITOFS_RTF_TARGET, st == ITOFS_TIER_OK ? "OK" : st == ITOFS_TIER_MARGINAL ? "MARGINAL" : "DEGRADED", s_degraded, s_sim ? " (SIMULATED TIMES)" : "");
    for (int i = 0; i < MAX_TIERS; i++) { if (rtf[i] >= 0) printf(" set%d %.3f", i, rtf[i]); else if (s_tier[i].valid) printf(" set%d not needed", i); else printf(" set%d absent", i); }
    printf("\n");
    if (st == ITOFS_TIER_MARGINAL)
        printf("NOTE: no weight set reaches RTF %.2f; the best (set %d) measures %.3f. Speech is planned with a start delay (gapless if the measurement holds), but there is little margin.\n", ITOFS_RTF_TARGET, pick, rtf[pick]);
    if (s_degraded)
        printf("WARNING: DEGRADED: even the fastest weight set runs at RTF %.3f (>= %.2f). Speech may stutter or fall behind on long sentences; the start delay grows with the sentence (capped at %.0f s). The firmware exposes this as degraded=1 in TIER_SELECT and `status`.\n",
               rtf[pick], ITOFS_RTF_DEGRADED, PLAN_MAX_DELAY_US / 1e6);
}

static void tier_status(void)
{
    printf("STATUS active_set=%d selected_set=%d degraded=%d status=%s start_delay=%s%s", s_active, s_selected, s_degraded,
           s_tstatus == ITOFS_TIER_OK ? "OK" : s_tstatus == ITOFS_TIER_MARGINAL ? "MARGINAL" : "DEGRADED", s_delay_override_ms >= 0 ? "fixed" : "planned", s_sim ? " SIMULATED_TIMES" : "");
    for (int i = 0; i < MAX_TIERS; i++) {
        if (!s_tier[i].valid) { printf(" | set%d ('%s'): absent", i, s_tier_label[i]); continue; }
        printf(" | set%d ('%s'): %u bytes", i, s_tier_label[i], (unsigned)s_tier[i].ext);
        if (s_tier[i].calibrated) printf(", measured RTF %.3f", s_tier[i].rtf); else printf(", not measured");
    }
    printf("\n");
}

static void halt(void);
static void synth_task(void *arg)
{
    (void)arg;
    int first = -1;
    for (int i = 0; i < MAX_TIERS && first < 0; i++) if (s_tier[i].valid) first = i;
    if (first < 0 || activate_tier(first)) { printf("ERROR no usable weight set\n"); halt(); }
    self_test(0);
    if (!s_selftest_ok) { printf("WARNING: the first weight set FAILED its self-test and is excluded; the other sets are tried below\n"); s_tier[first].valid = 0; }
#ifndef ITOFS_QEMU
    board_bench();                   // hardware: measure, keep the fastest weight-staging mode (QEMU: `bench` command)
#endif
    tiers_select();
    play_demos(-1);
    printf("READY (BOOT button = demos, serial: say <ids> | style <k> | demo [k] | test | stats | tier [k] | status | help)\n");
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
            icprof(s_req_tmp.n);
#endif
        } else if (s_req_tmp.kind == 8) {
            s_first = s_req_tmp.n;
            printf("OK first chunk %d frames%s\n", s_first, s_first >= s_chunk_frames || s_first <= 0 ? " (= a full chunk)" : "");
        } else if (s_req_tmp.kind == 6) {
            if (set_wmode(s_req_tmp.n)) printf("ERROR weight staging mode %d unavailable\n", s_req_tmp.n);
            else printf("OK weight staging %s\n", s3_wmode_name(s3_wmode));
        } else if (s_req_tmp.kind == 9) {            // tier [k]: activate weight set k (its self-test follows); no argument: status
            const int k = s_req_tmp.n;
            if (k < 0) tier_status();
            else if (k >= MAX_TIERS || !s_tier[k].valid) printf("ERROR weight set %d is absent\n", k);
            else if (activate_tier(k)) printf("ERROR could not activate set %d\n", k);
            else { s_selected = k; printf("OK weight set %d active\n", k); self_test(0); }
        } else if (s_req_tmp.kind == 10) {           // recal: measure again and re-select (with simulated times if `simtime` is on)
            for (int i = 0; i < MAX_TIERS; i++) s_tier[i].calibrated = 0;
            tiers_select();
        } else if (s_req_tmp.kind == 11) {
            tier_status();
        } else if (s_req_tmp.kind == 13) {
            ;
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
        s_in_req.kind = 7; s_in_req.n = atoi(line + 6);     // "icprof q" would be 0; "icprof 1" = quick (schedule 0 only: 1 core + 2 cores serialised)
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
#endif
    } else if (!strncmp(line, "tier", 4)) {          // tier: status; tier <k>: activate weight set k
        const char *a = line + 4;
        while (*a == ' ') a++;
        s_in_req.kind = 9; s_in_req.n = *a ? atoi(a) : -1;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "status", 6)) {
        s_in_req.kind = 11;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "recal", 5)) {
        s_in_req.kind = 10;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "simtime", 7)) {       // simtime off | simtime <rtf set0> <rtf set1> <rtf set2>: timing injection for the next `recal`
        const char *a = line + 7;
        while (*a == ' ') a++;
        s_in_req.kind = 13;                         // no-op request: only so that the console client sees READY
        if (!strncmp(a, "off", 3)) { s_sim = 0; printf("OK simulated timing off (the next recal measures)\n"); xQueueSend(s_req_q, &s_in_req, portMAX_DELAY); }
        else {
            char *e1, *e2; double r0 = strtod(a, &e1), r1 = strtod(e1, &e2), r2 = strtod(e2, NULL);
            if (e1 == a || r0 <= 0 || r1 <= 0 || r2 <= 0) { printf("ERROR simtime needs three positive RTFs: simtime 0.9 0.8 0.7\n"); return; }
            s_sim = 1; s_sim_rtf[0] = r0; s_sim_rtf[1] = r1; s_sim_rtf[2] = r2;
            printf("OK simulated timing on: set RTFs %.3f %.3f %.3f (the next recal uses them instead of the clock)\n", r0, r1, r2);
            xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
        }
    } else if (!strncmp(line, "unpack", 6)) {       // unpack pie | c: int4 row unpack by the PIE kernel or the C code (same results; for measurement)
        const char *a = line + 6;
        while (*a == ' ') a++;
        if (!strncmp(a, "c", 1)) { s3_w4_pie = 0; printf("OK int4 unpack: C code\n"); }
        else { s3_w4_pie = 1; printf("OK int4 unpack: PIE kernel (unchecked if the boot check failed!)\n"); }
        s_in_req.kind = 13; xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "first", 5)) {
        s_in_req.kind = 8; s_in_req.n = atoi(line + 5);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "delay", 5)) {
        if (!strncmp(line + 5, " auto", 5)) { s_delay_override_ms = -1; printf("OK start delay planned per utterance\n"); }
        else { s_delay_override_ms = atoi(line + 5); printf("OK start delay fixed at %d ms\n", s_delay_override_ms); }
        s_in_req.kind = 13; xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "wmode", 5)) {
        s_in_req.kind = 6; s_in_req.n = atoi(line + 5);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (line[0]) {
        printf("commands: say <comma-separated token ids> | style <0-%d> | act <8|16> | demo [0-%d] | test | stats | bench | "
               "wmode <0 direct|1 copy|2 gdma|3 gdma+prefetch> | first <frames of the first chunk> | delay <ms before playback starts | auto> | tier [k] | status | recal | simtime <rtf0 rtf1 rtf2 | off>\n", s_model.n_styles - 1, s_ndemo - 1);
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
#ifndef ITOFS_QEMU
        if (n <= 0) {            // found on the board: with only the native USB port connected the commands never arrived; read the USB-Serial/JTAG FIFO as well
            n = 0;
            while (n < (int)sizeof ch && usb_serial_jtag_ll_rxfifo_data_available()) n += usb_serial_jtag_ll_read_rxfifo(ch + n, sizeof ch - n);
        }
#endif
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
static void halt(void) { for (;;) vTaskDelay(1000); }

void app_main(void)
{
    printf("\n==== ItoFS TTS on ESP32-S3 ====\n");
    int nvalid = 0, first = -1;
    size_t max_ext = 0;
    for (int i = 0; i < MAX_TIERS; i++) {
        tier_t *T = &s_tier[i];
        T->p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)(0x40 + i), s_tier_label[i]);
        if (!T->p) { if (i == 0) printf("ERROR no 'weights' partition\n"); continue; }
        if (tier_ext(T->p, &T->ext, i > 0)) continue;
        T->valid = 1; nvalid++;
        if (first < 0) first = i;
        if (T->ext > max_ext) max_ext = T->ext;
    }
    if (!nvalid) halt();
    printf("weight sets in flash:");
    for (int i = 0; i < MAX_TIERS; i++) printf(" %d '%s' %s;", i, s_tier_label[i], s_tier[i].valid ? "present" : "absent");
    printf("\n");
    s_lim = (itofs_limits_t){ .max_tokens = MAX_TOKENS, .chunk_frames = STEADY_FRAMES, .act_bits = ITOFS_ACT_BITS };
    s_hb = s_bb = 0;
    for (int i = 0; i < MAX_TIERS; i++) {                // arena sizes: the largest over the sets (they share the arenas)
        if (!s_tier[i].valid) continue;
        size_t hb = 0, bb = 0; int hop = 0, sr = 0;
        if (tier_dims(i, &hb, &bb, &hop, &sr)) { s_tier[i].valid = 0; continue; }
        if (hb > s_hb) s_hb = hb;
        if (bb > s_bb) s_bb = bb;
        s_hop = hop; s_sr = sr;
    }
    // the active set's weights live in PSRAM next to the arenas (+ margin for the self-test buffer); else they run memory-mapped from flash (slow)
    {
        const size_t need = max_ext + 64 + s_bb + 320 * 1024;
        if (heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) > need) s_wbuf = heap_caps_aligned_alloc(64, max_ext + 64, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_wbuf) printf("weights: %u bytes do not fit in PSRAM next to the arenas: the active set will run memory-mapped from flash\n", (unsigned)max_ext);
    }
    s_chunk_frames = STEADY_FRAMES;
    s_chunk_samples = s_chunk_frames * s_hop;
    s_style = demo_style[0];   // replaced by the first blob demo's style once the demos are loaded
#ifndef ITOFS_QEMU
    i2s_setup();                     // before the arenas: DMA buffers must get internal RAM first
#else
    printf("QEMU build: I2S disabled (audio is generated and discarded)\n");
#endif
    // leave room in internal RAM for the task stacks, the staging tiles and the I2S slice (about 24 KB more than the arena itself)
    const char *hot_where = "internal SRAM";
    printf("internal SRAM before the hot arena: largest free block %u KB, free %u KB\n", (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) >> 10),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    // the arena needs one block of its size; what is left must still hold the task stacks, the 16 KB of staging tiles and the I2S slice
    if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) >= s_hb + 4 * 1024 && heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= s_hb + 56 * 1024)
        s_hot = heap_caps_aligned_alloc(16, s_hb, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_hot) { s_hot = heap_caps_aligned_alloc(16, s_hb, MALLOC_CAP_SPIRAM); hot_where = "PSRAM (internal SRAM too small)"; }
    s_bulk = heap_caps_aligned_alloc(16, s_bb, MALLOC_CAP_SPIRAM);
    if (!s_hot || !s_bulk) { printf("ERROR arena alloc (hot %u, bulk %u)\n", (unsigned)s_hb, (unsigned)s_bb); halt(); }
    printf("arena: hot %u bytes in %s, bulk %u bytes in PSRAM; chunk %d frames = %d samples; %d-bit activations\n", (unsigned)s_hb, hot_where,
           (unsigned)s_bb, s_chunk_frames, s_chunk_samples, s_lim.act_bits ? s_lim.act_bits : 8);
    s3_kernels_init();
    (void)first;

    s_free_q = xQueueCreate(NBUF, sizeof(buf_t));
    s_full_q = xQueueCreate(NBUF + 1, sizeof(buf_t));
    s_done_q = xQueueCreate(1, sizeof(buf_t));
    s_req_q = xQueueCreate(2, sizeof(req_t));
    for (int i = 0; i < NBUF; i++) {
        buf_t b = { heap_caps_malloc(sizeof(int16_t) * (size_t)s_chunk_samples, MALLOC_CAP_SPIRAM), 0 };   // PSRAM: the I2S driver copies from it
        if (!b.pcm) { printf("ERROR no PSRAM for PCM buffers\n"); halt(); }
        xQueueSend(s_free_q, &b, 0);
    }
    uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    mem_report("boot");
    xTaskCreatePinnedToCore(play_task, "itofs_play", 4096, NULL, 12, NULL, 1);
    xTaskCreatePinnedToCore(synth_task, "itofs_synth", 12288, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(console_task, "itofs_con", 4096, NULL, 4, NULL, 0);
    xTaskCreatePinnedToCore(button_task, "itofs_btn", 4096, NULL, 4, NULL, 0);
}
