// Ito on the ESP32-S3: StyleTTS 2 token ids in, 24 kHz audio out over I2S. Runs Ito v3 blobs (arch 3; the engine also
// still reads the older arch-2 format).
//
//   boot: load the weight blob (flash partition "weights") into PSRAM (or run it memory-mapped from flash when it does
//         not fit), run the self-test stored in the blob (the host C engine's PCM hash for golden sentence 0), run the
//         board benchmark (hardware build only: kernels + the demo sentences in every weight-staging mode, then keep the
//         fastest mode), then, only with CONFIG_ITOFS_BOOT_DEMOS, speak the demo sentences (stored in the blob; main/demo_sentences.h is the fallback).
//   BOOT button (GPIO0): speak the demo sentences again.
//   serial (UART0, 115200): "say 0,72,156,..."  speak token ids (tools/say.py)
//                           "style <0-63>" | "act <8|16>" | "demo" | "demo <k>" | "test" | "stats" | "bench" |
//                           "wmode <0|1|2>" (weight staging: direct / copy / gdma) | "first <n>" (frames in the
//                           first chunk) | "split [auto|row|full]" (GEMM split, see DUAL_CORE_POWER.md) | "equiv [light on|off]" (weight sets validated by ear) |
//                           "volume <0-100>" (ES8388 builds) | "cancel" (stop the
//                           utterance being spoken) | "rep <n> [k]" (n silent repetitions of the demos, with timing and hash) |
//                           "reboot" | "help"
//   I2S: sample rate from the blob (24 kHz), 16-bit, mono duplicated to stereo. Pins: BCLK GPIO15, LRCLK GPIO16, DOUT GPIO17 (plain DAC, default), or the ES8388 shield's.
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
#include "itofs_split.h"
#include "itofs_equiv.h"
#include "guard.h"
#ifdef CONFIG_ITOFS_AUDIO_ES8388
#include "es8388.h"
#endif

#ifdef CONFIG_ITOFS_AUDIO_ES8388       // the ESP_S3-AUDIN shield: ES8388 codec, I2C control, I2S with MCLK (esp32/README.md, "Audio output")
#define PIN_MCLK CONFIG_ITOFS_ES8388_PIN_MCLK
#define PIN_BCLK CONFIG_ITOFS_ES8388_PIN_BCLK
#define PIN_WS CONFIG_ITOFS_ES8388_PIN_WS
#define PIN_DOUT CONFIG_ITOFS_ES8388_PIN_DOUT
#else                                  // a plain I2S DAC / amplifier: BCLK GPIO15, LRCLK GPIO16, DOUT GPIO17, no MCLK
#define PIN_MCLK I2S_GPIO_UNUSED
#define PIN_BCLK 15
#define PIN_WS 16
#define PIN_DOUT 17
#endif
#define PIN_BOOT 0
#ifdef CONFIG_ITOFS_AUDIO_ES8388
#define VOLUME_HELP "volume <0-100> | "
#else
#define VOLUME_HELP ""
#endif
#ifdef CONFIG_ITOFS_AUDIO_ES8388
#define CODEC_MUTE(m) es8388_mute(m)       // the DAC is muted whenever nothing is spoken: a class-D amplifier makes no noise and no click on a silent, steady input
#else
#define CODEC_MUTE(m) ((void)0)
#endif
#define MAX_TOKENS 400
#define TAIL_SAMPLES 120             // end of an utterance: the last sample decays to zero over 5 ms (the engine's PCM ends at up to -45 dBFS, not at zero: a step would click)
#define I2S_SLICE 480                // samples per I2S write (20 ms = one DMA descriptor); a chunk (up to 300 ms) goes out in slices
#ifndef STEADY_FRAMES
#define STEADY_FRAMES 24             // frames per chunk once the start-up ramp is over (24 frames = 300 ms): every chunk reads each weight
                                     // matrix once from PSRAM, so bigger chunks mean less weight traffic per second of audio
#endif
#define NBUF 6                       // PCM chunk buffers between synthesis and I2S (PSRAM, 14 KB each): with a start delay the player waits while synthesis keeps going
                                     // (the start-delay planner models this pool: production of chunk k waits for the buffer of chunk k - NBUF)
// Start-up schedule (gapless speech from the first chunk). Time to first audio is the time of chunk 0; for playback never to starve, every later chunk
// must be ready before the audio of the chunks before it has played, and each chunk costs a fixed weight pass plus about 6.5 ms (central) per frame.
// A first chunk of 2 frames is out early but its 25 ms of audio is gone long before the 24-frame chunk after it is ready, so the speech has a gap unless
// playback is delayed. The ramp below starts with FIRST_FRAMES, then 11, 12, 14, 18 and 24, and the start delay is planned from the measured chunk times
// (itofs_sched.h), so playback never starves (measured: 0 underruns, esp32/BENCHMARKING.md). The text side advances in steps of 8 tokens meanwhile (a 24-token step would make one chunk 65 ms slower than its neighbours). The audio
// is bit-identical to any other chunking (host test G).
#ifndef RAMP_LIST
#define RAMP_LIST 11, 12, 14, 18     // frames of chunks 1, 2, ... after the first one, then STEADY_FRAMES
#endif
#ifndef TEXT_STEP_LIST
#define TEXT_STEP_LIST 8, 8, 8, 8, 8, 8, 8, 8   // text-side tokens added per step in chunk 0, 1, ... (0 or past the list: the engine's default, 24)
#endif
#ifndef FIRST_FRAMES
#define FIRST_FRAMES 10              // frames of the first chunk (10 = 125 ms of audio). `first <n>` changes it (n >= chunk = no ramp at all); a smaller first chunk is out sooner but the
                                     // planner delays playback more (esp32/BENCHMARKING.md, "first chunk": 2 frames gave an underrun)
#endif
#ifndef START_DELAY_MS
#define START_DELAY_MS -1            // -1: the start delay is planned per utterance from the boot calibration (itofs_sched.h); >= 0 fixes it in ms (the `delay <ms>` command does the same at run time)
#endif
#ifndef ITOFS_ACT_BITS
#define ITOFS_ACT_BITS 8             // activations of the int8-weight layers: 8 (one int8 plane, default) or 16; `act` command
#endif
// Initial policy of the GEMM split (the stored record takes over after the first boot): menuconfig "Ito > Initial split policy"; the compile definition
// S3_GEMM_DUAL_DEFAULT=0 / 1 (ITOFS_EXTRA_DEFS) forces row / full.
#if defined(S3_GEMM_DUAL_DEFAULT)
#define SPLIT_POLICY (S3_GEMM_DUAL_DEFAULT ? ITOFS_PREF_FULL : ITOFS_PREF_ROW)
#elif defined(CONFIG_ITOFS_SPLIT_ROW)
#define SPLIT_POLICY ITOFS_PREF_ROW
#elif defined(CONFIG_ITOFS_SPLIT_FULL)
#define SPLIT_POLICY ITOFS_PREF_FULL
#else
#define SPLIT_POLICY ITOFS_PREF_AUTO
#endif
#define WDT_RUN_MS 15000             // the hardware watchdog resets the chip when no synthesis loop has fed it for this long (a chunk takes about 0.25 s)
#define WDT_TEST_MS 4000             // the same during the supply test (a pass of the test takes about 50 ms)
#define SUPPLY_TEST_MS 3000          // long supply test (first boot, `split auto`)
#define SUPPLY_CHECK_MS 250          // quick re-check of a passed verdict at every boot

static itofs_model_t s_model;
static const void *s_blob; static size_t s_blob_size;     // the active weight set
static void *s_hot, *s_bulk; static size_t s_hb, s_bb;      // the engine's hot (internal SRAM) and bulk (PSRAM) arenas
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
static volatile int s_cancel, s_speaking;   // `cancel`: set by the console while an utterance is spoken; the synthesis stops at the next chunk, the player drops what it holds and flushes the I2S ring
static volatile int64_t s_cancel_t;         // esp_timer time of the cancel command
static volatile int64_t s_utt_t0;       // esp_timer time (us) at which the current utterance was handed to the engine
static int s_delay_override_ms = START_DELAY_MS;   // >= 0: hold the DAC until this long after the text arrived, whatever the plan says (`delay <ms>`); -1: use the plan (`delay auto`)
static volatile int64_t s_cur_delay_us;            // the planned start delay of the current utterance (from its text arrival); raised while the player waits if the chunks come slower than calibrated
static int s_trace;                                // `trace 1`: print the chunk times of every utterance
static int s_active = -1, s_selected = -1, s_degraded;   // weight set in use / chosen by the boot calibration / degraded flag (see the weight-set section)
#define PLAN_MAX_DELAY_US 4000000.0
static double s_obs_us[ITOFS_SCHED_MAXCH];         // production times of the chunks of the current utterance so far

typedef struct { int kind; int n; int16_t tok[MAX_TOKENS]; } req_t;   // 0 say, 1 demos, 2 self-test, 3 style, 4 act bits, 5 bench, 6 wmode, 7 icprof, 8 first, 9 tier, 10 recal, 11 status, 13 no-op, 14 rep, 15 split, 16 hang (test builds)
static req_t s_req_tmp;

static double ms(int64_t us) { return us / 1000.0; }

static void mem_report(const char *tag)
{
    size_t pt = heap_caps_get_total_size(MALLOC_CAP_SPIRAM), pmin = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    size_t it = heap_caps_get_total_size(MALLOC_CAP_INTERNAL), imin = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    printf("MEM %s: peak PSRAM used %u / %u KB, peak internal SRAM used %u / %u KB | free now: PSRAM %u KB, internal SRAM %u KB\n", tag,
           (unsigned)((pt - pmin) / 1024), (unsigned)(pt / 1024), (unsigned)((it - imin) / 1024), (unsigned)(it / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
}

// ------------------------------------------------------------------------------------------------------------------
// audio output
// ------------------------------------------------------------------------------------------------------------------
#ifndef ITOFS_QEMU
static i2s_chan_handle_t s_tx;
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
// Test-only microphone loopback (see the `loopback` command below): the player records the level of what it hands to the I2S, in 10 ms frames, from the first slice on.
static i2s_chan_handle_t s_rx;
#define LB_ENV_MAX 2400                       // 24 s of 10 ms frames
static volatile int s_lb_on;
static volatile int64_t s_lb_first_us;        // esp_timer time at which the first slice of the utterance was handed to the I2S
static float s_lb_env[LB_ENV_MAX];            // RMS of the played PCM per 10 ms frame, dBFS
static int s_lb_env_n, s_lb_acc_n;
static double s_lb_acc;
static void lb_play_note(const int16_t *pcm, int n)
{
    if (s_lb_env_n == 0 && s_lb_acc_n == 0) s_lb_first_us = esp_timer_get_time();
    for (int i = 0; i < n; i++) {
        s_lb_acc += (double)pcm[i] * pcm[i];
        if (++s_lb_acc_n == 240 && s_lb_env_n < LB_ENV_MAX) { s_lb_env[s_lb_env_n++] = (float)(10.0 * log10(s_lb_acc / 240 + 1.0) - 90.3); s_lb_acc = 0; s_lb_acc_n = 0; }
        else if (s_lb_acc_n == 240) { s_lb_acc = 0; s_lb_acc_n = 0; }
    }
}
#endif
static void i2s_setup(void)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = 6;
    cc.dma_frame_num = 480;
    cc.auto_clear = true;            // silence (not a repeated buffer) on underrun
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
    ESP_ERROR_CHECK(i2s_new_channel(&cc, &s_tx, &s_rx));
#else
    ESP_ERROR_CHECK(i2s_new_channel(&cc, &s_tx, NULL));
#endif
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(24000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = { .mclk = PIN_MCLK, .bclk = PIN_BCLK, .ws = PIN_WS, .dout = PIN_DOUT, .din = I2S_GPIO_UNUSED,
                      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false } },
    };
    sc.clk_cfg.sample_rate_hz = (uint32_t)s_sr;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx, &sc));
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
    sc.gpio_cfg.dout = I2S_GPIO_UNUSED; sc.gpio_cfg.din = CONFIG_ITOFS_ES8388_PIN_DIN;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &sc));          // the receive channel shares the transmit channel's clocks; it is enabled only during a `loopback`
#endif
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx));
#ifdef CONFIG_ITOFS_AUDIO_ES8388
    const int e = es8388_init(CONFIG_ITOFS_ES8388_PIN_SDA, CONFIG_ITOFS_ES8388_PIN_SCL, CONFIG_ITOFS_AUDIO_VOLUME);   // the clocks run, so the codec can lock to them
    if (e) printf("ERROR ES8388 codec %s: nothing will be heard\n", e == -1 ? "does not answer on the I2C bus" : "bus error");
    else { printf("audio: ES8388 codec, MCLK GPIO%d, BCLK GPIO%d, LRCK GPIO%d, DOUT GPIO%d, output volume %d %%, DAC muted until the first utterance\n", PIN_MCLK, PIN_BCLK, PIN_WS, PIN_DOUT, CONFIG_ITOFS_AUDIO_VOLUME); }
#endif
}

// Cancel: drop the audio queued in the DMA ring (about 120 ms): stop the channel, fill the ring with silence, start again.
static void cancel_flush(int16_t *st)
{
    memset(st, 0, sizeof(int16_t) * 2 * (size_t)I2S_SLICE);
    i2s_channel_disable(s_tx);
    for (int i = 0; i < 6; i++) { size_t n = 0; i2s_channel_preload_data(s_tx, st, sizeof(int16_t) * 2 * (size_t)I2S_SLICE, &n); }
    i2s_channel_enable(s_tx);
    CODEC_MUTE(1);
    printf("CANCEL: audio flushed %.1f ms after the command\n", ms(esp_timer_get_time() - s_cancel_t));
}
#else
static void cancel_flush(int16_t *st) { (void)st; }
#endif

static void play_task(void *arg)
{
    (void)arg;
    int16_t *st = heap_caps_malloc(sizeof(int16_t) * 2 * (size_t)I2S_SLICE + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);   // one 20 ms slice
    buf_t b;
    int16_t last = 0;                // the last sample handed to the I2S in this utterance
    int64_t end_us = 0;              // when the audio handed to the I2S so far has been played out (the DMA ring holds ~120 ms, and i2s_channel_write returns once the data is queued, not played)
    for (;;) {
        int starved = 0;
        if (s_playing && xQueueReceive(s_full_q, &b, 0) != pdTRUE) {
            starved = 1;             // the queue is empty: an underrun only if the DAC has also run out of audio (checked below; found on the board: the ring made every late chunk look like a gap)
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        } else if (!s_playing) {
            xQueueReceive(s_full_q, &b, portMAX_DELAY);
        }
        if (b.n == 0) {
#ifndef ITOFS_QEMU
            if (s_playing && !s_cancel && last) {          // end of the utterance: a short ramp from the last sample to zero instead of a step
                for (int i = 0; i < TAIL_SAMPLES; i++) st[2 * i] = st[2 * i + 1] = (int16_t)((int32_t)last * (TAIL_SAMPLES - i) / TAIL_SAMPLES);
                size_t wr = 0;
                i2s_channel_write(s_tx, st, sizeof(int16_t) * 2 * TAIL_SAMPLES, &wr, portMAX_DELAY);
            }
#endif
            if (s_playing) {                               // the audio handed to the I2S has played out at end_us (+ the tail ramp); then the codec is muted
                const int64_t w = end_us + (int64_t)TAIL_SAMPLES * 1000000 / s_sr + 60000 - esp_timer_get_time();
                if (w > 0) vTaskDelay(pdMS_TO_TICKS((w + 999) / 1000));
                CODEC_MUTE(1);
            }
            last = 0; s_playing = 0; xQueueSend(s_done_q, &b, portMAX_DELAY); continue;
        }
        if (s_cancel) {                                    // cancelled: chunks still arriving are dropped until the end marker
            if (s_playing) { cancel_flush(st); s_playing = 0; }
            xQueueSend(s_free_q, &b, portMAX_DELAY);
            continue;
        }
        if (!s_playing) {                                  // pre-roll: the first chunk waits for the delay, later chunks keep arriving meanwhile
            CODEC_MUTE(0);                                 // (the codec's volume ramp runs during the pre-roll, which is at least one chunk long)
            for (;;) {                                     // (re-read in <= 10 ms steps: the synthesis task may raise the planned delay)
                if (s_cancel) break;
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
        for (int o = 0; o < b.n && !s_cancel; o += I2S_SLICE) {
            const int m = b.n - o < I2S_SLICE ? b.n - o : I2S_SLICE;
            for (int i = 0; i < m; i++) { st[2 * i] = b.pcm[o + i]; st[2 * i + 1] = b.pcm[o + i]; }
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
            if (s_lb_on) lb_play_note(b.pcm + o, m);
#endif
            size_t wr = 0;
            i2s_channel_write(s_tx, st, sizeof(int16_t) * 2 * (size_t)m, &wr, portMAX_DELAY);
        }
        last = b.pcm[b.n - 1];
        if (s_cancel) { cancel_flush(st); s_playing = 0; }
#else
        (void)st; (void)last;
#endif
        xQueueSend(s_free_q, &b, portMAX_DELAY);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// synthesis
// ------------------------------------------------------------------------------------------------------------------
typedef struct { double ttfa_ms, begin_ms, compute_ms, audio_s; int frames, tok_first; double plan_delay_ms; int plan_underruns, plan_feasible; uint32_t hash; int cancelled, chunks; } stats_t;

// Synthesise tokens; stream chunks to the player (play=1) or into out[] (play=0).
static int tier_plan(int n_tok, int nobs, itofs_plan_t *pl);       // below: start-delay plan from the active weight set's calibration
static int speak(const int *tok, int n, int style, uint32_t seed, int play, int16_t *out, long out_max, stats_t *S)
{
    S->tok_first = 0; S->plan_delay_ms = -1; S->plan_underruns = 0; S->plan_feasible = 1; S->hash = 2166136261u; S->cancelled = 0; S->chunks = 0;
    s_cur_delay_us = 0;
    s_cancel = 0; s_speaking = play;
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
        guard_wdt_feed();
        if (play) xQueueReceive(s_free_q, &b, portMAX_DELAY);
        if (s_cancel) { if (play) xQueueSend(s_free_q, &b, portMAX_DELAY); break; }
        int16_t *dst = play ? b.pcm : out + pos;
        s_ctx.text_step = text_step_for(kch);
        const int want = chunk_frames_for(kch++) * s_hop;                                  // low-latency first chunk, ramp, steady state
        int room = play ? want : (int)((out_max - pos) < want ? (out_max - pos) : want);
        int64_t a = esp_timer_get_time();
        int got = room >= s_hop ? itofs_next_chunk(&s_ctx, dst, room) : 0;
        int64_t e = esp_timer_get_time();
        compute += e - a;
        if (got <= 0) { if (play) xQueueSend(s_free_q, &b, portMAX_DELAY); break; }
        S->chunks++;
        if (kch - 1 < ITOFS_SCHED_MAXCH) s_obs_us[kch - 1] = (double)(e - a);
        if (ttfa < 0) { ttfa = e - t0; S->tok_first = s_ctx.n_done; }
        pos += got;
        for (int i = 0; i < got; i++) { S->hash ^= (uint16_t)dst[i]; S->hash *= 16777619u; }
        if (play) {
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
    S->cancelled = s_cancel; s_cancel = 0; s_speaking = 0;
    if (s_trace) {                                   // `trace 1`: the time of every chunk of this utterance
        printf("CHUNKS %d (ms):", S->chunks);
        for (int i = 0; i < S->chunks && i < ITOFS_SCHED_MAXCH; i++) printf(" %.1f", s_obs_us[i] / 1e3);
        printf("\n");
    }
    s_ctx.text_step = 0;
    S->begin_ms = ms(t1 - t0); S->ttfa_ms = ms(ttfa); S->compute_ms = ms(compute);
    S->frames = s_ctx.T; S->audio_s = (double)pos / s_sr; (void)T;
    return (int)pos;
}

static void print_timing(const char *what, const stats_t *S, int n_tok, int style)
{
    if (S->cancelled) printf("CANCELLED %s after %.2f s of audio\n", what, S->audio_s);
    printf("TIMING %s: %d tokens, style %d, %.2f s audio | first chunk %.1f ms (begin %.1f ms; %s text side, %d tokens final at first chunk) | "
           "compute %.1f ms, RTF %.3f | underruns %d (start delay %s %.0f ms%s) | weight set %d | split %s | weight staging %s: %ld GEMM calls, %ld announced ahead, %ld not | %d chunks, %.1f MB of weights read | pcm fnv32 %08lx\n",
           what, n_tok, style, S->audio_s, S->ttfa_ms, S->begin_ms, s_ctx.text_incr ? "incremental" : "whole-sentence", S->tok_first,
           S->compute_ms, S->compute_ms / 1000.0 / S->audio_s, s_underruns, s_delay_override_ms >= 0 ? "fixed" : "planned",
           s_delay_override_ms >= 0 ? (double)s_delay_override_ms : S->plan_delay_ms, s_delay_override_ms >= 0 || S->plan_feasible ? "" : ", model predicts underruns", s_active, s3_gemm_dual ? "full" : "row", s3_wmode_name(s3_wmode),
           s3_stats.calls, s3_stats.pf_hit, s3_stats.pf_miss + s3_stats.pf_free, S->chunks, s3_stats.wbytes / 1e6, (unsigned long)S->hash);
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

// ------------------------------------------------------------------------------------------------------------------
// GEMM split: row (safe) or full (both cores in the GEMM). The decision, the supply test and the persistence are described in itofs_split.h
// and in DUAL_CORE_POWER.md; s3_gemm_dual is what the kernels read.
// ------------------------------------------------------------------------------------------------------------------
static itofs_split_rec_t s_split;
static const char *s_rst_name = "?";
static itofs_split_plan_t s_plan;               // what the record asked for at this boot (set in app_main, applied by the synthesis task)
static int s_split_changed;                   // the split changed after the weight sets were measured: the measurements must be redone

static void split_save(void)
{
    if (guard_rec_save(&s_split)) printf("WARNING: the split record could not be stored in NVS (the verdict will not survive a reset)\n");
}

static void split_print(void)
{
    const itofs_split_rec_t *r = &s_split;
    printf("SPLIT mode=%s pref=%s verdict=%s why=\"%s\" policy=%s | boots %u, tests %u (passed %u, failed %u), brownout resets %u, watchdog/panic resets %u, hash mismatches %u, interrupted tests %u | last reset: %s\n",
           s3_gemm_dual ? "full" : "row", itofs_split_pref_name(r->pref), itofs_split_verdict_name(r->verdict), itofs_split_why_name(r->why), itofs_split_pref_name(SPLIT_POLICY),
           (unsigned)r->boots, (unsigned)r->tests, (unsigned)r->passes, (unsigned)r->fails, (unsigned)r->brownouts, (unsigned)r->wdt_resets, (unsigned)r->mismatches, (unsigned)r->interrupted, s_rst_name);
}

// The supply test on the active weights; the watchdog is tightened around it, so a hang or a collapse resets the board and the next boot reads the record.
static int supply_test(int dur_ms)
{
#ifdef ITOFS_QEMU
    (void)dur_ms;
    printf("SUPPLY_TEST skipped: QEMU has no PIE current draw, counted as PASS (the dual-core path is exercised by the self-test)\n");
    return 1;
#else
    long passes = 0;
    guard_wdt_arm(WDT_TEST_MS);
    const int64_t t0 = esp_timer_get_time();
    const int ok = s3_supply_test(s_blob, s_blob_size, s_hot, s_hb, dur_ms, &passes);
    guard_wdt_arm(WDT_RUN_MS);
    printf("SUPPLY_TEST %s: full split on the engine's shapes, %ld passes in %.0f ms, every result %s the one-core result\n", dur_ms == SUPPLY_TEST_MS ? "long" : "quick", passes,
           ms(esp_timer_get_time() - t0), ok ? "equal to" : "DIFFERENT FROM");
    return ok;
#endif
}

// Apply what the record asks for: run the test it asks for, store the outcome, set the split.
static void split_apply(itofs_split_plan_t p)
{
    int full = p.full;
    if (p.save) split_save();                     // before the test: a reset during it must find testing = 1
    if (p.test == 2) { full = itofs_split_test_done(&s_split, supply_test(SUPPLY_TEST_MS)); split_save(); }
    else if (p.test == 1 && full && !supply_test(SUPPLY_CHECK_MS)) { itofs_split_fail(&s_split, ITOFS_WHY_RUN_MISMATCH); full = 0; split_save(); }
    if (full != s3_gemm_dual) s_split_changed = 1;
    s3_gemm_dual = full;
    split_print();
}

// Evidence against the full split while it runs: latch the row split.
static void split_latch_row(itofs_why_t why)
{
    itofs_split_fail(&s_split, why);
    split_save();
    s3_gemm_dual = 0; s_split_changed = 1;
    printf("SPLIT: %s -> the row split is latched; `split auto|full` re-enables the full split\n", itofs_split_why_name(why));
    split_print();
}

static void self_test(int both)
{
    for (int attempt = 0; attempt < 2; attempt++) {
        s_selftest_ok = 1;
        if (!s_st) { printf("SELFTEST skipped: this weight blob has no self-test record\n"); s_tested = 1; return; }
        self_test_one(s_st, both, 1);
        if (both && s_st16) self_test_one(s_st16, 0, 0);
        s_tested = 1;
        if (s_selftest_ok || !s3_gemm_dual) return;
        split_latch_row(ITOFS_WHY_RUN_MISMATCH);                  // the same test in the row split decides between a bad split and a bad weight set
    }
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

#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
// ------------------------------------------------------------------------------------------------------------------
// Test-only microphone loopback (CONFIG_ITOFS_AUDIO_LOOPBACK): the shield's microphones sit next to the speaker and go to the ES8388 ADC. `loopback <k> [dump]` records
// the codec ADC on the I2S receive channel (same clocks as the DAC, so the two timelines are sample-accurate relative to each other) while demo sentence k is spoken, and
// reports the microphone's noise floor, DC offset, clipping, level, the delay from the first sample handed to the DAC to the first sound at the microphone, and how well the
// microphone's 10 ms level follows the played level (dropouts: played loud, microphone silent; clicks: played silent, microphone jumps). `dump` prints the samples as hex.
// ------------------------------------------------------------------------------------------------------------------
static int s_mic_sel, s_mic_gain = 4;
typedef struct { int16_t *buf; size_t cap_frames; volatile size_t got_frames; volatile int stop, done; } lb_cap_t;
static void lb_rx_task(void *arg)
{
    lb_cap_t *c = arg;
    while (!c->stop && c->got_frames + 480 <= c->cap_frames) {
        size_t n = 0;
        if (i2s_channel_read(s_rx, c->buf + 2 * c->got_frames, 480 * 4, &n, 100) == ESP_OK) c->got_frames += n / 4;
    }
    c->done = 1;
    vTaskDelete(NULL);
}

static double lb_db(const int16_t *x, size_t from, size_t n, int ch, double *dc, int *peak)   // rms in dBFS of channel ch over n frames of the interleaved buffer
{
    double s = 0, sum = 0; int pk = 0;
    for (size_t i = from; i < from + n; i++) { const int v = x[2 * i + ch]; s += (double)v * v; sum += v; if (abs(v) > pk) pk = abs(v); }
    if (dc) *dc = sum / (double)n;
    if (peak) *peak = pk;
    return 10.0 * log10(s / (double)n + 1.0) - 90.3;
}

static void loopback_run(int k, int dump)
{
    const size_t fs = (size_t)s_sr, cap = fs * 17, warm = fs * 5 / 2;     // 2.5 s for the ADC's DC offset to settle, then the quiet microphone, then the utterance
    lb_cap_t c = { heap_caps_malloc(cap * 4, MALLOC_CAP_SPIRAM), cap, 0, 0, 0 };
    if (!c.buf) { printf("LOOPBACK ERROR: no memory\n"); return; }
    if (es8388_adc_init(s_mic_sel, s_mic_gain)) { printf("LOOPBACK ERROR: the codec ADC could not be configured\n"); heap_caps_free(c.buf); return; }
    s_lb_env_n = 0; s_lb_acc_n = 0; s_lb_acc = 0; s_lb_on = 1;
    i2s_channel_enable(s_rx);
    const int64_t t_rx0 = esp_timer_get_time();
    xTaskCreatePinnedToCore(lb_rx_task, "itofs_rx", 4096, &c, 7, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(3300));                                 // 2.5 s of settling, then the microphone alone: noise floor, DC offset
    demo_tokens(k, s_tok_buf);
    stats_t S;
    speak(s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, 1, NULL, 0, &S);
    vTaskDelay(pdMS_TO_TICKS(900));
    c.stop = 1;
    while (!c.done) vTaskDelay(1);
    i2s_channel_disable(s_rx);
    s_lb_on = 0;
    const size_t got = c.got_frames, n0 = (size_t)((s_lb_first_us - t_rx0) * (int64_t)fs / 1000000), nplay = (size_t)s_lb_env_n * 240;
    printf("LOOPBACK demo %d: %u frames recorded, first DAC sample at frame %u (%.1f ms after the receive channel started), %d played frames of 10 ms\n", k, (unsigned)got, (unsigned)n0, (s_lb_first_us - t_rx0) / 1e3, s_lb_env_n);
    if (n0 < warm + 4800 || n0 + nplay + 4800 > got) { printf("LOOPBACK ERROR: the recording does not cover the utterance\n"); heap_caps_free(c.buf); return; }
    int best = 0; double best_lvl = -200;
    for (int ch = 0; ch < 2; ch++) {
        double dc, dcs; int pk, pks;
        const double nf = lb_db(c.buf, warm, n0 - warm - 480, ch, &dc, &pk), lvl = lb_db(c.buf, n0, nplay, ch, &dcs, &pks);
        size_t clip = 0; for (size_t i = warm; i < got; i++) if (abs(c.buf[2 * i + ch]) >= 32700) clip++;
        printf("LOOPBACK mic %c (input pair %d, PGA %d dB): noise floor %.1f dBFS rms, DC offset %.1f LSB, during speech: rms %.1f dBFS, peak %d (%.1f dBFS), clipped samples %u\n", ch ? 'R' : 'L', s_mic_sel, s_mic_gain * 3,
               nf, dc, lvl, pks, 20.0 * log10(pks / 32768.0 + 1e-9), (unsigned)clip);
        if (lvl > best_lvl) { best_lvl = lvl; best = ch; }
    }
    // 10 ms levels of the better channel, from the first DAC sample on
    const int nfr = (int)((got - n0) / 240) - 1;
    float *mic = malloc(sizeof(float) * (size_t)nfr);
    if (!mic) { heap_caps_free(c.buf); return; }
    for (int j = 0; j < nfr; j++) mic[j] = (float)lb_db(c.buf, n0 + (size_t)j * 240, 240, best, NULL, NULL);
    double pmax = -200; for (int j = 0; j < s_lb_env_n; j++) if (s_lb_env[j] > pmax) pmax = s_lb_env[j];
    double nf_db = lb_db(c.buf, warm, n0 - warm - 480, best, NULL, NULL);
    int d_best = 0; double r_best = -2;
    for (int d = 0; d <= 60 && d + s_lb_env_n < nfr; d++) {          // lag of the microphone behind the DAC, in 10 ms frames
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0; int n = 0;
        for (int j = 0; j < s_lb_env_n; j++) if (s_lb_env[j] > pmax - 40) { const double x = s_lb_env[j], y = mic[j + d]; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; n++; }
        const double r = (n * sxy - sx * sy) / (sqrt(n * sxx - sx * sx) * sqrt(n * syy - sy * sy) + 1e-9);
        if (r > r_best) { r_best = r; d_best = d; }
    }
    int drop = 0, loud = 0, click = 0, quiet = 0;
    for (int j = 0; j < s_lb_env_n && j + d_best < nfr; j++) {
        if (s_lb_env[j] > pmax - 25) { loud++; if (mic[j + d_best] < nf_db + 6) drop++; }
        else if (s_lb_env[j] < pmax - 50) { quiet++; if (mic[j + d_best] > nf_db + 15) click++; }
    }
    size_t first = 0;                                                  // first 1 ms window above the noise floor + 12 dB (two in a row)
    for (size_t i = n0; i + 48 < got && !first; i += 24)
        if (lb_db(c.buf, i, 24, best, NULL, NULL) > nf_db + 12 && lb_db(c.buf, i + 24, 24, best, NULL, NULL) > nf_db + 12) first = i;
    printf("LOOPBACK result (mic %c): the microphone's level follows the played level with correlation %.2f at a lag of %d ms; first sound at the microphone %.1f ms after the first DAC sample; "
           "dropouts %d of %d loud frames; clicks %d of %d silent frames; speech %.1f dB above the noise floor\n", best ? 'R' : 'L', r_best, d_best * 10, first ? (double)(first - n0) * 1000.0 / (double)fs : -1.0, drop, loud, click, quiet, best_lvl - nf_db);
    if (dump) {
        printf("LBDATA_BEGIN rate %d samples %u first_dac_sample %u channel %c\n", s_sr, (unsigned)got, (unsigned)n0, best ? 'R' : 'L');
        for (size_t i = 0; i < got; i += 32) {
            printf("LBDATA");
            for (size_t j = i; j < i + 32 && j < got; j++) printf(" %04x", (uint16_t)c.buf[2 * j + best]);
            printf("\n");
            guard_wdt_feed();
        }
        printf("LBDATA_END\n");
    }
    free(mic);
    heap_caps_free(c.buf);
}
#endif

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
                guard_wdt_feed();
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
// Only one is in PSRAM at a time. At boot the firmware activates them in that order (all of them are measured), synthesises a representative sentence (the blob's self-test sentence)
// without playback with the shipped chunk schedule, records the real time of every chunk (compute plus weight fetch from PSRAM), turns that into a measured
// real-time factor. Among the sets whose RTF is <= 0.85 it keeps the one with the lowest planned start delay (itofs_sched.h). If none is, it keeps the fastest and says so; if that one is still
// >= 0.95 it prints a WARNING line and sets the degraded flag instead of silently stuttering. The same measured chunk times plan, per utterance, the
// playback start delay (an adaptive jitter pre-buffer: the first audio is released as soon as the model says the speaker cannot run dry).
// `simtime a b c` replaces the measured chunk times by modelled ones (RTF a, b, c for the three sets) so the whole logic can be exercised in QEMU.
// ------------------------------------------------------------------------------------------------------------------
#define MAX_TIERS 3
typedef struct {
    const esp_partition_t *p; size_t ext; int valid;
    int calibrated; double rtf, rtf_steady, rtf_whole; itofs_cal_t cal;
    int blocks; long n_int4; char desc[48];
    uint32_t id; int has_id;            // the blob's self-test hash: identifies voice and set (itofs_equiv.h)
} tier_t;
static tier_t s_tier[MAX_TIERS];
static const char *const s_tier_label[MAX_TIERS] = { "weights", "weights_b", "weights_c" };
static itofs_tier_status_t s_tstatus = ITOFS_TIER_OK;
static uint8_t *s_wbuf;                       // PSRAM copy of the active set (NULL: running memory-mapped from flash)
static esp_partition_mmap_handle_t s_map_h; static int s_mapped;
static itofs_limits_t s_lim;
static int s_sim; static double s_sim_rtf[MAX_TIERS] = { 0.9, 0.8, 0.7 };

static int tier_ext(const esp_partition_t *p, size_t *ext, uint32_t *id, int *has_id, int quiet)
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
    *has_id = 0;
    for (uint32_t i = 0; i < nt; i++) {
        uint8_t ent[104];
        if (esp_partition_read(p, 16 + cfg_len + (size_t)i * 104, ent, sizeof ent) != ESP_OK) return -1;
        uint32_t off, nb, so, dt, out;
        memcpy(&dt, ent + 64, 4); memcpy(&out, ent + 76, 4); memcpy(&off, ent + 88, 4); memcpy(&nb, ent + 92, 4); memcpy(&so, ent + 96, 4);
        if (off + nb > e) e = off + nb;
        if (!strncmp((const char *)ent, "selftest", 9) && dt == 2 && nb >= 28) {       // the record's word 5 is the host engine's PCM hash: the set's identity
            uint32_t w[6];
            if (esp_partition_read(p, off, w, sizeof w) == ESP_OK && w[0] == 0x32525453u) { *id = w[5]; *has_id = 1; }
        }
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
            guard_wdt_feed();
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
        guard_wdt_feed();
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
    if (s_trace) {
        printf("CALIB_CHUNKS set %d (frames:ms):", i);
        for (int k = 0; k < kc; k++) printf(" %d:%.1f", c->frames[k], c->t_us[k] / 1e3);
        printf(" | text-side begin %.1f ms\n", c->begin_us / 1e3);
    }
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

// Sets judged equivalent by ear (per voice, itofs_equiv.h: female main int8 and int4, male main, int4 and light) are chosen among on speed alone: the lowest planned
// start delay, then the lowest first chunk (itofs_sched_pick). A set outside the list (the female light set until it is approved, or an unknown blob) is used only when no
// listed set meets the RTF target. Approval of the light sets: CONFIG_ITOFS_APPROVE_LIGHT at build time, or `equiv light on` (stored in NVS).
#ifdef CONFIG_ITOFS_APPROVE_LIGHT
#define APPROVE_BUILD ITOFS_APPROVE_LIGHT
#else
#define APPROVE_BUILD 0u
#endif
static unsigned s_approve_nvs;                 // override stored in NVS (`equiv`)
static unsigned approve_mask(void) { return APPROVE_BUILD | s_approve_nvs; }
static int tier_equivalent(int i) { return s_tier[i].has_id && itofs_set_equivalent(s_tier[i].id, approve_mask()); }
static const char *tier_who(int i, char *buf, size_t n)
{
    const itofs_set_info_t *si = s_tier[i].has_id ? itofs_set_lookup(s_tier[i].id) : NULL;
    if (si) snprintf(buf, n, "%s %s, %s", si->voice, si->set, tier_equivalent(i) ? (si->validated ? "validated by ear" : "approved by override") : "not validated by ear");
    else snprintf(buf, n, "unknown blob, not validated by ear");
    return buf;
}
static void tiers_select(void)
{
    itofs_set_score_t sc[MAX_TIERS];
    itofs_tier_status_t st = ITOFS_TIER_OK;
    int pick = -1;
    for (int attempt = 0; attempt < MAX_TIERS; attempt++) {
        s_split_changed = 0;
        for (int i = 0; i < MAX_TIERS; i++) sc[i] = (itofs_set_score_t){ -1, 0, 0, s_tier[i].valid && tier_equivalent(i) };
        for (int i = 0; i < MAX_TIERS; i++) {                 // measure every set (the choice needs their start delays), in the split in force
            if (!s_tier[i].valid) continue;
            if (s_active != i && activate_tier(i)) { s_tier[i].valid = 0; continue; }
            calibrate_tier(i);
            sc[i].rtf = s_tier[i].rtf;
            if (s_tier[i].calibrated) {
                itofs_plan_t pl;
                itofs_sched_plan(&s_tier[i].cal, s_tier[i].cal.n_tok, 0, NBUF, ITOFS_SCHED_SAFETY, PLAN_MAX_DELAY_US, NULL, 0, &pl);
                sc[i].start_delay_us = pl.start_delay_us; sc[i].first_chunk_us = s_tier[i].cal.begin_us + s_tier[i].cal.t_us[0];
            }
        }
        pick = itofs_sched_pick(sc, MAX_TIERS, ITOFS_RTF_TARGET, ITOFS_RTF_DEGRADED, &st);
        if (pick < 0) { printf("TIER_SELECT ERROR: no weight set could be measured\n"); s_degraded = 1; return; }
        if (pick != s_active && activate_tier(pick)) { s_tier[pick].valid = 0; continue; }
        if (!s_tested) {                                      // the chosen set must reproduce the host engine's PCM bit for bit
            self_test(0);
            if (s_split_changed) continue;                    // the full split failed here: measure everything again in the row split
            if (!s_selftest_ok) { printf("TIER_SELECT: weight set %d FAILED its self-test and is excluded\n", pick); s_tier[pick].valid = 0; s_tested = 0; continue; }
        }
        break;
    }
    s_selected = pick; s_tstatus = st; s_degraded = st == ITOFS_TIER_DEGRADED;
    printf("TIER_SELECT chosen=%d ('%s', %s) measured_rtf=%.3f target=%.2f status=%s degraded=%d split=%s%s | measured (RTF, planned start delay for %d tokens, first chunk):", pick, s_tier[pick].p->label, s_tier[pick].desc, sc[pick].rtf,
           ITOFS_RTF_TARGET, st == ITOFS_TIER_OK ? "OK" : st == ITOFS_TIER_MARGINAL ? "MARGINAL" : "DEGRADED", s_degraded, s3_gemm_dual ? "full" : "row", s_sim ? " (SIMULATED TIMES)" : "",
           s_tier[pick].cal.n_tok);
    for (int i = 0; i < MAX_TIERS; i++) {
        char who[64];
        if (sc[i].rtf >= 0) printf(" set%d %.3f, %.0f ms, %.0f ms (%s);", i, sc[i].rtf, sc[i].start_delay_us / 1e3, sc[i].first_chunk_us / 1e3, tier_who(i, who, sizeof who));
        else printf(" set%d %s;", i, s_tier[i].valid ? "not measured" : "absent");
    }
    printf("\n");
    if (!sc[pick].equivalent)
        printf("NOTE: the chosen set %d is not on the list of sets validated by ear for its voice (`equiv`): no validated set meets RTF %.2f in this split.\n", pick, ITOFS_RTF_TARGET);
    if (st == ITOFS_TIER_MARGINAL)
        printf("NOTE: no weight set reaches RTF %.2f; the best (set %d) measures %.3f. Speech is planned with a start delay (gapless if the measurement holds), but there is little margin.\n", ITOFS_RTF_TARGET, pick, sc[pick].rtf);
    if (s_degraded)
        printf("WARNING: DEGRADED: even the fastest weight set runs at RTF %.3f (>= %.2f). Speech may stutter or fall behind on long sentences; the start delay grows with the sentence (capped at %.0f s). The firmware exposes this as degraded=1 in TIER_SELECT and `status`.\n",
               sc[pick].rtf, ITOFS_RTF_DEGRADED, PLAN_MAX_DELAY_US / 1e6);
}

static void tier_status(void)
{
    printf("STATUS split=%s active_set=%d selected_set=%d degraded=%d status=%s start_delay=%s%s", s3_gemm_dual ? "full" : "row", s_active, s_selected, s_degraded,
           s_tstatus == ITOFS_TIER_OK ? "OK" : s_tstatus == ITOFS_TIER_MARGINAL ? "MARGINAL" : "DEGRADED", s_delay_override_ms >= 0 ? "fixed" : "planned", s_sim ? " SIMULATED_TIMES" : "");
    for (int i = 0; i < MAX_TIERS; i++) {
        if (!s_tier[i].valid) { printf(" | set%d ('%s'): absent", i, s_tier_label[i]); continue; }
        printf(" | set%d ('%s'): %u bytes", i, s_tier_label[i], (unsigned)s_tier[i].ext);
        if (s_tier[i].calibrated) printf(", measured RTF %.3f", s_tier[i].rtf); else printf(", not measured");
    }
    printf("\n");
}

// `rep <n> [k]`: n silent repetitions of the demo sentences (all, or only k): first chunk, RTF and the PCM hash of every run (the hash must not change), then
// mean (sd) [min..max] per sentence and the memory before and after (a leak shows as a difference).
#define REP_MAX_SAMPLES (12 * 24000)
static void repeat_demos(int n, int only)
{
    int16_t *pcm = heap_caps_malloc(sizeof(int16_t) * REP_MAX_SAMPLES, MALLOC_CAP_SPIRAM);
    if (!pcm) { printf("REP ERROR: no memory\n"); return; }
    double sf[MAX_DEMOS] = {0}, sff[MAX_DEMOS] = {0}, sr[MAX_DEMOS] = {0}, srr[MAX_DEMOS] = {0};
    double f_min[MAX_DEMOS], f_max[MAX_DEMOS] = {0}, r_min[MAX_DEMOS], r_max[MAX_DEMOS] = {0};
    uint32_t h0[MAX_DEMOS] = {0};
    int bad[MAX_DEMOS] = {0}, cnt[MAX_DEMOS] = {0};
    for (int k = 0; k < MAX_DEMOS; k++) f_min[k] = r_min[k] = 1e30;
    const size_t p0 = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) + sizeof(int16_t) * REP_MAX_SAMPLES, i0 = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    for (int r = 0; r < n; r++)
        for (int k = 0; k < s_ndemo; k++) {
            if (only >= 0 && k != only) continue;
            demo_tokens(k, s_tok_buf);
            stats_t S;
            const int got = speak(s_tok_buf, s_demo_len[k], s_demo_style[k], 1u + (uint32_t)k, 0, pcm, REP_MAX_SAMPLES, &S);
            if (got <= 0) continue;
            const double rtf = S.compute_ms / 1000.0 / S.audio_s;
            if (!cnt[k]) h0[k] = S.hash;
            const int ok = S.hash == h0[k];
            bad[k] += !ok;
            printf("REP %d demo %d: %d tokens, first chunk %.1f ms, RTF %.3f, pcm fnv32 %08lx %s\n", r + 1, k, s_demo_len[k], S.ttfa_ms, rtf, (unsigned long)S.hash, ok ? "(same)" : "(DIFFERENT)");
            sf[k] += S.ttfa_ms; sff[k] += S.ttfa_ms * S.ttfa_ms; sr[k] += rtf; srr[k] += rtf * rtf; cnt[k]++;
            if (S.ttfa_ms < f_min[k]) f_min[k] = S.ttfa_ms;
            if (S.ttfa_ms > f_max[k]) f_max[k] = S.ttfa_ms;
            if (rtf < r_min[k]) r_min[k] = rtf;
            if (rtf > r_max[k]) r_max[k] = rtf;
        }
    for (int k = 0; k < s_ndemo; k++) {
        if (!cnt[k]) continue;
        const double c = cnt[k], mf = sf[k] / c, mr = sr[k] / c;
        printf("REP_SUMMARY demo %d (%d tokens, split %s, set %d): n %d | first chunk %.1f ms (sd %.2f) [%.1f..%.1f] | RTF %.3f (sd %.4f) [%.3f..%.3f] | pcm fnv32 %08lx, %d runs different\n", k, s_demo_len[k],
               s3_gemm_dual ? "full" : "row", s_active, cnt[k], mf, sqrt(fmax(0.0, sff[k] / c - mf * mf)), f_min[k], f_max[k], mr, sqrt(fmax(0.0, srr[k] / c - mr * mr)), r_min[k], r_max[k], (unsigned long)h0[k], bad[k]);
    }
    heap_caps_free(pcm);
    printf("REP memory: free PSRAM %u -> %u KB, free internal SRAM %u -> %u KB\n", (unsigned)(p0 / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
           (unsigned)(i0 / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
}

static void halt(void);
static void synth_task(void *arg)
{
    (void)arg;
    int first = -1;
    for (int i = 0; i < MAX_TIERS && first < 0; i++) if (s_tier[i].valid) first = i;
    if (first < 0 || activate_tier(first)) { printf("ERROR no usable weight set\n"); halt(); }
    split_apply(s_plan);                 // the supply test (if due) runs on the first weight set, before anything else uses both cores
    self_test(0);
    if (!s_selftest_ok) { printf("WARNING: the first weight set FAILED its self-test and is excluded; the other sets are tried below\n"); s_tier[first].valid = 0; }
#ifndef ITOFS_QEMU
    board_bench();                   // hardware: measure, keep the fastest weight-staging mode (QEMU: `bench` command)
#endif
    tiers_select();
#ifdef CONFIG_ITOFS_BOOT_DEMOS
    play_demos(-1);
#endif
    printf("READY (BOOT button = demos, serial: say <ids> | style <k> | demo [k] | test | stats | tier [k] | status | help)\n");
    for (;;) {
        while (xQueueReceive(s_req_q, &s_req_tmp, pdMS_TO_TICKS(1000)) != pdTRUE) guard_wdt_feed();      // idle: the watchdog is fed once a second
        guard_wdt_feed();
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
        } else if (s_req_tmp.kind == 14) {
            repeat_demos(s_req_tmp.n >> 8, (s_req_tmp.n & 0xFF) - 1);
        #ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
        } else if (s_req_tmp.kind == 17) {           // loopback <k> [dump]
            loopback_run(s_req_tmp.n & 0xFF, s_req_tmp.n >> 8);
#endif
        } else if (s_req_tmp.kind == 18) {           // chunk <frames>: steady-state chunk size (at most the arena's STEADY_FRAMES); the weight sets must be measured again (`recal`)
            s_chunk_frames = s_req_tmp.n; s_chunk_samples = s_chunk_frames * s_hop;
            printf("OK steady chunk %d frames (%d ms of audio); `first %d` or more turns the start-up ramp off\n", s_chunk_frames, s_chunk_frames * s_hop * 1000 / s_sr, s_chunk_frames);
        } else if (s_req_tmp.kind == 19) {           // equiv [light on|off]
            if (s_req_tmp.n >= 0) {
                s_approve_nvs = s_req_tmp.n ? ITOFS_APPROVE_LIGHT : 0;
                if (guard_u8_save(NVS_KEY_APPROVE, (uint8_t)s_approve_nvs)) printf("WARNING: the override could not be stored in NVS\n");
            }
            for (int i = 0; i < MAX_TIERS; i++) {
                char who[64];
                if (s_tier[i].valid) printf("EQUIV set %d ('%s'): %s\n", i, s_tier_label[i], tier_who(i, who, sizeof who));
            }
            printf("EQUIV override: light sets %s (build option %s, stored override %s)\n", approve_mask() & ITOFS_APPROVE_LIGHT ? "approved" : "not approved",
                   APPROVE_BUILD ? "on" : "off", s_approve_nvs ? "on" : "off");
            if (s_req_tmp.n >= 0) { for (int i = 0; i < MAX_TIERS; i++) s_tier[i].calibrated = 0; tiers_select(); }
        } else if (s_req_tmp.kind == 15) {           // split auto|row|full
            split_apply(itofs_split_set_pref(&s_split, (itofs_pref_t)s_req_tmp.n));
            if (s3_gemm_dual) self_test(0);          // the full split must reproduce the host engine's PCM before it is trusted
            if (s_split_changed) tiers_select();     // the measurements of the weight sets belong to the split they were made in
#ifdef ITOFS_INJECT_HANG
        } else if (s_req_tmp.kind == 16) {           // test builds only: hang the synthesis task, with interrupts on (n = 0) or off (n = 1)
            printf("HANG injected (interrupts %s): the watchdog must reset the board within %d s\n", s_req_tmp.n ? "off" : "on", WDT_RUN_MS / 1000);
            fflush(stdout);
            if (s_req_tmp.n) portDISABLE_INTERRUPTS();
            for (;;) ;
#endif
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
    } else if (!strncmp(line, "cancel", 6)) {        // handled here, not by the synthesis task: it is busy with the utterance being cancelled
        if (s_speaking) { s_cancel_t = esp_timer_get_time(); s_cancel = 1; printf("OK cancel\n"); }
        else printf("OK nothing to cancel\n");
    } else if (!strncmp(line, "equiv", 5)) {         // equiv: show which sets are validated by ear; equiv light on|off: approve the light sets of every voice (stored in NVS), then measure again
        const char *a = line + 5;
        while (*a == ' ') a++;
        s_in_req.kind = 19;
        if (!*a) s_in_req.n = -1;
        else if (!strncmp(a, "light on", 8)) s_in_req.n = 1;
        else if (!strncmp(a, "light off", 9)) s_in_req.n = 0;
        else { printf("ERROR equiv [light on|off]\n"); return; }
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "split", 5)) {         // split: show; split auto|row|full: set the preference (stored in NVS)
        const char *a = line + 5;
        while (*a == ' ') a++;
        if (!*a) split_print();
        else if (!strncmp(a, "auto", 4) || !strncmp(a, "row", 3) || !strncmp(a, "full", 4)) {
            s_in_req.kind = 15; s_in_req.n = a[0] == 'a' ? ITOFS_PREF_AUTO : a[0] == 'r' ? ITOFS_PREF_ROW : ITOFS_PREF_FULL;
            xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
        } else printf("ERROR split auto|row|full\n");
#ifdef CONFIG_ITOFS_AUDIO_ES8388
    } else if (!strncmp(line, "volume", 6)) {         // volume <0-100>: codec output volume in percent (91 = 0 dB, 100 = +4.5 dB); 0 mutes the DAC
        const int v = atoi(line + 6);
        if (v < 0 || v > 100 || !line[6]) { printf("ERROR volume 0-100\n"); return; }
        es8388_set_volume_percent(v);                // 0 is a true mute that utterances do not lift (the codec's volume register alone only reaches -45 dB)
        printf("OK volume %d\n", v);
#endif
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
    } else if (!strncmp(line, "loopback", 8)) {      // loopback <k> [dump]
        char *e; const long k = strtol(line + 8, &e, 10);
        if (k < 0 || k >= s_ndemo) { printf("ERROR loopback <demo 0-%d> [dump]\n", s_ndemo - 1); return; }
        s_in_req.kind = 17; s_in_req.n = (int)k | (strstr(e, "dump") ? 256 : 0);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "micsel", 6)) {        // micsel 0-3: the codec's input (single-ended pair 1 or 2, differential pair 1 or 2)
        s_mic_sel = atoi(line + 6) & 3; printf("OK input pair %d (applies to the next loopback)\n", s_mic_sel);
    } else if (!strncmp(line, "micgain", 7)) {       // micgain 0-8: PGA gain in steps of 3 dB
        s_mic_gain = atoi(line + 7); if (s_mic_gain < 0) s_mic_gain = 0; if (s_mic_gain > 8) s_mic_gain = 8; printf("OK PGA gain %d dB (applies to the next loopback)\n", s_mic_gain * 3);
#endif
    } else if (!strncmp(line, "chunk", 5)) {
        const int n = atoi(line + 5);
        if (n < 2 || n > STEADY_FRAMES) { printf("ERROR chunk 2-%d\n", STEADY_FRAMES); return; }
        s_in_req.kind = 18; s_in_req.n = n;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "cores", 5)) {           // cores 1|2: one core for everything, or both (the split decides the GEMM); for measurement
        s3_dual_enabled = atoi(line + 5) != 1; printf("OK %d core%s\n", s3_dual_enabled ? 2 : 1, s3_dual_enabled ? "s" : "");
    } else if (!strncmp(line, "trace", 5)) {           // trace 0|1: print the time of every chunk
        s_trace = atoi(line + 5) != 0; printf("OK chunk trace %s\n", s_trace ? "on" : "off");
    } else if (!strncmp(line, "rep", 3)) {            // rep <n> [k]
        char *e; const long n = strtol(line + 3, &e, 10);
        const long k = strtol(e, NULL, 10);
        if (n < 1 || n > 1000) { printf("ERROR rep <n 1-1000> [demo]\n"); return; }
        s_in_req.kind = 14; s_in_req.n = (int)(n << 8) | (*e && k >= 0 && k < s_ndemo ? (int)k + 1 : 0);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "reboot", 6)) {
        printf("rebooting\n");
        vTaskDelay(pdMS_TO_TICKS(100));
        esp_restart();
#ifdef ITOFS_INJECT_HANG
    } else if (!strncmp(line, "hang", 4)) {
        s_in_req.kind = 16; s_in_req.n = strstr(line, "irq") != NULL;
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
#endif
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
        else {
            int d = atoi(line + 5);
            if (d < 0) d = 0;
            if (d > (int)(PLAN_MAX_DELAY_US / 1000)) d = (int)(PLAN_MAX_DELAY_US / 1000);   // a longer hold would starve the watchdog (the synthesis waits for the player's buffers)
            s_delay_override_ms = d; printf("OK start delay fixed at %d ms (at most %d)\n", d, (int)(PLAN_MAX_DELAY_US / 1000));
        }
        s_in_req.kind = 13; xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (!strncmp(line, "wmode", 5)) {
        s_in_req.kind = 6; s_in_req.n = atoi(line + 5);
        xQueueSend(s_req_q, &s_in_req, portMAX_DELAY);
    } else if (line[0]) {
        printf("commands: say <comma-separated token ids> | style <0-%d> | act <8|16> | demo [0-%d] | test | stats | bench | "
               "wmode <0 direct|1 copy|2 gdma|3 gdma+prefetch> | first <frames of the first chunk> | chunk <steady frames> | cores <1|2> | trace <0|1> | split [auto|row|full] | equiv [light on|off] | " VOLUME_HELP "cancel | rep <n> [demo] | reboot | delay <ms before playback starts | auto> | tier [k] | status | recal | simtime <rtf0 rtf1 rtf2 | off>\n", s_model.n_styles - 1, s_ndemo - 1);
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
    guard_wdt_arm(WDT_RUN_MS);
    {   // GEMM split: the stored verdict and the evidence of the reset that led here (see itofs_split.h); the row split runs until the supply test has passed
        const itofs_rst_t rst = guard_reset_class(&s_rst_name);
        if (guard_rec_load(&s_split)) { itofs_split_default(&s_split, SPLIT_POLICY); printf("SPLIT: no valid record in NVS, starting a new one (policy %s)\n", itofs_split_pref_name(SPLIT_POLICY)); }
        s_plan = itofs_split_boot(&s_split, SPLIT_POLICY, rst);
        { uint8_t a = 0; if (!guard_u8_load(NVS_KEY_APPROVE, &a)) s_approve_nvs = a & ITOFS_APPROVE_LIGHT; }
        s3_gemm_dual = 0;
    }
    int nvalid = 0, first = -1;
    size_t max_ext = 0;
    for (int i = 0; i < MAX_TIERS; i++) {
        tier_t *T = &s_tier[i];
        T->p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)(0x40 + i), s_tier_label[i]);
        if (!T->p) { if (i == 0) printf("ERROR no 'weights' partition\n"); continue; }
        if (tier_ext(T->p, &T->ext, &T->id, &T->has_id, i > 0)) continue;
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
#ifdef ITOFS_KCHECK
    heap_caps_free(s_hot); heap_caps_free(s_wbuf);   // the check needs the internal SRAM of the engine's hot arena and 4 MB of PSRAM; a check build does no synthesis
    guard_wdt_arm(120000);
    s3_kernel_check(1);
    halt();
#endif
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
