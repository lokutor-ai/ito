// int8 x int8 -> int32 GEMM for the ItoFS engine on the ESP32-S3, with weight staging into internal SRAM.
//
// The engine needs the exact int32 accumulator (every frame has its own activation scale, which is what makes
// streaming bit-exact without calibration), so this does not use esp-nn's requantising fully-connected kernels.
// It uses esp-nn's S3 PIE/SIMD dot primitive (esp_nn_dot_s8_unaligned_esp32s3, 16 int8 MACs per step), weight-
// stationary: a tile of output channels is reused across every frame (and both activation planes) of the chunk, and
// the output channels are split across both cores for large layers. The engine pads the activation row stride to 16
// bytes, so the primitive's first operand is always aligned; weight rows may be at any alignment. Input widths that are
// not a multiple of 16 finish the last <16 products in plain C. The result is exact (bit-identical) in every mode.
//
// Weight staging (s3_wmode), for weights in PSRAM (the blob is copied there at boot):
//   0 direct : the dot products read the weights in place through the 64 KB data cache
//   1 copy   : memcpy a tile (<= TILE_BYTES) PSRAM -> internal SRAM, then compute it from SRAM
//   2 gdma   : the GDMA copies tile i+1 PSRAM -> SRAM while the CPU computes tile i (double buffer per core). Needs the
//              boot self-test of the async-memcpy channel (QEMU does not emulate it: there the copies are emulated, see below).
//   3 gdma+  : mode 2 plus cross-call prefetch. The engine announces the weights of the GEMM call that FOLLOWS (itofs_qnext_fn, before
//              the current call is made); the tiles of the next call are then fetched into the buffers the current call is not using
//              any more, during its last tile and during the float work (rescale, GELU, LayerNorm, ...) between the calls, instead of
//              being waited for at the start of the next call. Results are bit-identical (only WHEN the weights arrive changes).
// Under QEMU (ITOFS_EMU_DMA) the GDMA does not exist: the same control flow runs with a software copy done when the tile is waited for
// (the buffer holds 0xA5 until then), so a tile used before its copy completed, or a buffer reused too early, changes the audio.
// Weights in internal RAM or memory-mapped flash always use 0.
#include "kernels_s3.h"
#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_timer.h"
#include "esp_async_memcpy.h"
#include "esp_cache.h"
#include "common_functions.h"   // esp-nn: esp_nn_dot_s8_unaligned_esp32s3
#include "itofs.h"

#define TOC 8
#define DUAL_MIN_MACS 32768
#define TILE_BYTES 4096         // per staging buffer (2 per core): 16 KB of internal SRAM in total (was 10 KB per buffer: the double-buffered pipeline loses
                                // under 2 % with 4 KB tiles, trace_sim.py, and the 24 KB saved pay for WIDE_ROWS = 12)
#define TILE_SLACK 128          // GDMA moves 64-byte aligned supersets of the tile

volatile int s3_dual_enabled = 1;
volatile int s3_wmode = 0;
volatile int s3_dma_errors = 0;
s3_stats_t s3_stats;

#define PACK_BYTES 1280          // one packed weight row: the widest GEMM input is 1202 (80 vectors of 16 B)
typedef struct { const int8_t *w; int in, out, rows, rb; } pcall_t;     // a GEMM call (the shared weights base, width, output channels, rows, bytes per weight row: in for int8, the int4 row size for int4)
typedef struct { const int8_t *tp; size_t n; const int8_t *src; } slot_t;
#define MAXQ 3
typedef struct {
    int8_t *buf[2]; async_memcpy_handle_t dma; SemaphoreHandle_t sem; int8_t *pk;
    // weight stream of this core: the calls it will work on (cq[0] = running / next), tiles fetched / used / completions taken
    pcall_t cq[MAXQ]; int ncq;
    int gen_i, gen_t;                  // next tile to fetch: tile gen_t of call cq[gen_i]
    long fetched, consumed, taken;
    slot_t slot[2];                    // buffer (seq & 1) holds tile `seq`
#ifdef ITOFS_EMU_DMA
    struct { int8_t *dst; const void *src; size_t n; } emu[4]; int emu_n;
#endif
} core_ctx_t;
static core_ctx_t g_cc[2];
static int g_dma_ok, g_tiles_ok;

typedef struct { const int8_t *x; int rows, ldx, in; const int8_t *w; int out; int32_t *acc; int oc0, oc1; int mode; int rb, w4; } job_t;   // rb: bytes per weight row; w4: int4 rows

const char *s3_wmode_name(int m) { return m == 0 ? "direct" : m == 1 ? "copy" : m == 2 ? "gdma" : "gdma+prefetch"; }
int s3_dma_ok(void) { return g_dma_ok; }
void s3_stats_reset(void) { memset(&s3_stats, 0, sizeof s3_stats); }

// main/dot_rows_s3.S: one aligned weight row against every activation row (2 instructions per 16 MACs)
extern void itofs_s3_dot_rows(const int8_t *x, int ldx, const int8_t *w, int rows_len16, int32_t *acc, int stride_bytes);
// copy nv16 x 16 bytes from any address into a 16-byte aligned buffer (reads up to 16 bytes past the end)
extern void itofs_s3_pack_row(int8_t *dst, const int8_t *src, int nv16);

extern void itofs_s3_w4_unpack(int8_t *dst, const uint8_t *row, int ng);    // main/dot_rows_s3.S: PIE unpack, 18 instructions per group of 32 weights
volatile int s3_w4_pie = 0;               // 1: int4 rows are unpacked by the PIE kernel (set at boot if it reproduces the C unpack exactly), 0: by the C code below

// int4 row -> int8 (the engine's itofs_w4_unpack_row on aligned 32-bit words; row and dst 16-byte aligned): a group's 16 packed bytes give
// 32 weights as  q * m + (128 - zp * m) per byte, x ^ 0x80 = (q - zp) * m  (see itofs.h). Writes ng * 32 bytes.
static inline void w4_unpack_row(const uint8_t *row, int in, int8_t *dst)
{
    const int ng = (in + 31) >> 5;
    const uint8_t *prm = row + ng * 16;
    const uint32_t *nb = (const uint32_t *)row;
    uint32_t *d = (uint32_t *)dst;
    for (int g = 0; g < ng; g++) {
        const uint32_t m = prm[g] >> 4, zp = prm[g] & 15u, C = (128u - zp * m) * 0x01010101u;
        const uint32_t w0 = nb[0], w1 = nb[1], w2 = nb[2], w3 = nb[3];
        d[0] = (((w0 & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u; d[1] = (((w1 & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u;
        d[2] = (((w2 & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u; d[3] = (((w3 & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u;
        d[4] = ((((w0 >> 4) & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u; d[5] = ((((w1 >> 4) & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u;
        d[6] = ((((w2 >> 4) & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u; d[7] = ((((w3 >> 4) & 0x0F0F0F0Fu) * m) + C) ^ 0x80808080u;
        nb += 4; d += 8;
    }
}

// channels [o0, o1) whose weights start at wt (row stride in; int4: j->rb bytes per packed row)
static inline void compute(const job_t *j, const int8_t *wt, int o0, int o1, int8_t *pk)
{
    const int in = j->in, in16 = in >> 4, r0 = in16 << 4;
    if (j->w4) {       // each packed row is unpacked once into the aligned scratch row (amortised over every activation row), then the aligned dot kernel runs on it
        const int nv = (in + 15) >> 4;
        for (int o = o0; o < o1; o++) {
            if (s3_w4_pie) itofs_s3_w4_unpack(pk, (const uint8_t *)wt + (size_t)(o - o0) * j->rb, (in + 31) >> 5);
            else w4_unpack_row((const uint8_t *)wt + (size_t)(o - o0) * j->rb, in, pk);
            itofs_s3_dot_rows(j->x, j->ldx, pk, (j->rows << 16) | nv, j->acc + o, j->out * 4);
        }
        return;
    }
    if ((in & 15) == 0 && ((uintptr_t)wt & 15) == 0 && j->rows < 65536) {   // every weight row aligned: fast kernel
        for (int o = o0; o < o1; o++)
            itofs_s3_dot_rows(j->x, j->ldx, wt + (size_t)(o - o0) * in, (j->rows << 16) | in16, j->acc + o, j->out * 4);
        return;
    }
    if (pk && j->rows < 65536 && in <= PACK_BYTES - 32) {
        // input width not a multiple of 16 (or unaligned weights): the engine zero-pads every activation row up to ldx, so one more
        // vector of zeros adds nothing. Each weight row is first shifted into an aligned buffer (3 instructions per 16 bytes, once for
        // all rows of the chunk), then the aligned kernel runs over every row (2 instructions per 16 bytes instead of 4 plus a scalar tail).
        const int nv = (in + 15) >> 4;
        for (int o = o0; o < o1; o++) {
            itofs_s3_pack_row(pk, wt + (size_t)(o - o0) * in, nv);
            itofs_s3_dot_rows(j->x, j->ldx, pk, (j->rows << 16) | nv, j->acc + o, j->out * 4);
        }
        return;
    }
    for (int t = o0; t < o1; t += TOC) {
        const int te = (t + TOC < o1) ? t + TOC : o1;
        for (int r = 0; r < j->rows; r++) {
            const int8_t *xr = j->x + (size_t)r * j->ldx;
            int32_t *ar = j->acc + (size_t)r * j->out;
            for (int o = t; o < te; o++) {
                const int8_t *wr = wt + (size_t)(o - o0) * in;
                int32_t a = in16 ? esp_nn_dot_s8_unaligned_esp32s3(xr, wr, in16) : 0;
                for (int i = r0; i < in; i++) a += (int32_t)xr[i] * wr[i];
                ar[o] = a;
            }
        }
    }
}

static bool IRAM_ATTR dma_done_cb(async_memcpy_handle_t h, async_memcpy_event_t *e, void *arg)
{
    (void)h; (void)e;
    BaseType_t hp = pdFALSE;
    xSemaphoreGiveFromISR((SemaphoreHandle_t)arg, &hp);
    return hp == pdTRUE;
}

// ---- the copy engine: GDMA (hardware) or a deferred software copy (QEMU) -------------------------------------------
static void hw_wait_one(core_ctx_t *c);
// start copying n bytes (src 64-byte aligned superset) into dst; completions are signalled in issue order on c->sem
static void hw_issue(core_ctx_t *c, int8_t *dst, const void *src, size_t n)
{
#ifdef ITOFS_EMU_DMA
    memset(dst, 0xA5, n);                       // poison: the tile is only valid once it has been waited for
    c->emu[c->emu_n].dst = dst; c->emu[c->emu_n].src = src; c->emu[c->emu_n].n = n; c->emu_n++;
#else
    if (esp_async_memcpy(c->dma, dst, (void *)src, n, dma_done_cb, c->sem) != ESP_OK) {
        s3_dma_errors++;                                    // (not reached with <= 2 transfers in flight, backlog 4) software copy instead,
        while (c->taken < c->fetched) { hw_wait_one(c); c->taken++; }     // after the earlier copies, so completions stay in issue order
        memcpy(dst, src, n);
        xSemaphoreGive(c->sem);
    }
#endif
}
// wait for the oldest outstanding copy
static void hw_wait_one(core_ctx_t *c)
{
#ifdef ITOFS_EMU_DMA
    if (c->emu_n) {
        memcpy(c->emu[0].dst, c->emu[0].src, c->emu[0].n);
        for (int i = 1; i < c->emu_n; i++) c->emu[i - 1] = c->emu[i];
        c->emu_n--;
    }
#else
    if (xSemaphoreTake(c->sem, pdMS_TO_TICKS(500)) != pdTRUE) s3_dma_errors++;
#endif
}

// start copying the 64-byte aligned superset of [src, src + n) into buffer bi; returns where the tile starts in it (the bench's single copies)
static const int8_t *dma_issue(core_ctx_t *c, int bi, const int8_t *src, size_t n)
{
    const uintptr_t a = (uintptr_t)src & ~(uintptr_t)63, e = ((uintptr_t)src + n + 63) & ~(uintptr_t)63;
    hw_issue(c, c->buf[bi], (const void *)a, e - a);
    return c->buf[bi] + ((uintptr_t)src - a);
}
static void dma_wait(core_ctx_t *c) { hw_wait_one(c); }

// ---- the weight stream ---------------------------------------------------------------------------------------------
// output channels of the call that core k works on: [*c0, *c1)
static void core_range(const pcall_t *p, int k, int *c0, int *c1)
{
    const int dual = s3_dual_enabled && (long)p->rows * p->in * p->out >= DUAL_MIN_MACS && p->out >= 2 * TOC;
    const int split = dual ? ((p->out / 2) + TOC - 1) / TOC * TOC : p->out;
    if (k == 0) { *c0 = 0; *c1 = split; } else { *c0 = split; *c1 = dual ? p->out : split; }
}
static inline int tile_channels(int rb) { const int t = TILE_BYTES / rb; return t < 1 ? 1 : t; }    // in <= 1536 always (the engine's widest GEMM input is 1202)

// issue fetches while a buffer is free
static void pump(int k)
{
    core_ctx_t *c = &g_cc[k];
    while (c->fetched - c->consumed < 2) {
        int c0 = 0, c1 = 0, nt = 0;
        while (c->gen_i < c->ncq) {
            core_range(&c->cq[c->gen_i], k, &c0, &c1);
            const int toc = tile_channels(c->cq[c->gen_i].rb);
            nt = (c1 - c0 + toc - 1) / toc;
            if (c->gen_t < nt) break;
            c->gen_i++; c->gen_t = 0;
        }
        if (c->gen_i >= c->ncq) return;
        const pcall_t *p = &c->cq[c->gen_i];
        const int toc = tile_channels(p->rb), t0 = c0 + c->gen_t * toc, t1 = (t0 + toc < c1) ? t0 + toc : c1;
        const int8_t *src = p->w + (size_t)t0 * p->rb;
        const size_t n = (size_t)(t1 - t0) * p->rb;
        const uintptr_t a = (uintptr_t)src & ~(uintptr_t)63, e = ((uintptr_t)src + n + 63) & ~(uintptr_t)63;
        slot_t *sl = &c->slot[c->fetched & 1];
        hw_issue(c, c->buf[c->fetched & 1], (const void *)a, e - a);
        sl->tp = c->buf[c->fetched & 1] + ((uintptr_t)src - a); sl->n = n; sl->src = src;
        c->fetched++; c->gen_t++;
    }
}
// wait until tile `seq` has arrived
static void wait_tile(core_ctx_t *c, long seq) { while (c->taken <= seq) { hw_wait_one(c); c->taken++; } }
// all outstanding copies done, stream emptied
static void drain(int k)
{
    core_ctx_t *c = &g_cc[k];
    wait_tile(c, c->fetched - 1);
    c->consumed = c->fetched;
    c->ncq = 0; c->gen_i = 0; c->gen_t = 0;
}

// this core's part of a staged call (mode 2 / 3): the call is cq[0]
static void gemm_stream(const job_t *j, int k)
{
    core_ctx_t *c = &g_cc[k];
    const pcall_t *p = &c->cq[0];
    int c0 = 0, c1 = 0;
    core_range(p, k, &c0, &c1);
    const int toc = tile_channels(p->rb);
    for (int t0 = c0; t0 < c1; t0 += toc) {
        const int t1 = (t0 + toc < c1) ? t0 + toc : c1;
        pump(k);                                   // this tile and the next one (or the next call's first) in flight
        const long seq = c->consumed;
        wait_tile(c, seq);
        compute(j, c->slot[seq & 1].tp, t0, t1, c->pk);
        c->consumed++;
        pump(k);                                   // the freed buffer starts on the next tile / call at once
    }
    c->ncq--;                                      // the call is done
    for (int i = 0; i < c->ncq; i++) c->cq[i] = c->cq[i + 1];
    if (c->gen_i > 0) c->gen_i--; else c->gen_t = 0;
}

static void gemm_range(const job_t *j, core_ctx_t *c)
{
    if (j->oc1 <= j->oc0) return;
    const int rb = j->rb;
    if (j->mode == 0) { compute(j, j->w + (size_t)j->oc0 * rb, j->oc0, j->oc1, c->pk); return; }
    if (j->mode == 1) {
        const int toc = tile_channels(rb);
        for (int t0 = j->oc0; t0 < j->oc1; t0 += toc) {
            const int t1 = (t0 + toc < j->oc1) ? t0 + toc : j->oc1;
            memcpy(c->buf[0], j->w + (size_t)t0 * rb, (size_t)(t1 - t0) * rb);
            compute(j, c->buf[0], t0, t1, c->pk);
        }
        return;
    }
    gemm_stream(j, c == &g_cc[0] ? 0 : 1);
}

// ---- stream entry / hints ----------------------------------------------------------------------------------------------
static void stream_pop(int k)
{
    core_ctx_t *c = &g_cc[k];
    c->ncq--;
    for (int i = 0; i < c->ncq; i++) c->cq[i] = c->cq[i + 1];
    if (c->gen_i > 0) c->gen_i--; else c->gen_t = 0;
}
static inline int same_call(const pcall_t *a, const pcall_t *b) { return a->w == b->w && a->in == b->in && a->out == b->out && a->rows == b->rows && a->rb == b->rb; }
// a staged call starts: its tiles must head the stream (they usually were hinted already); otherwise the stream is flushed
static void stream_enter(const pcall_t *p)
{
    for (int k = 0; k < 2; k++) {
        core_ctx_t *c = &g_cc[k];
        if (c->ncq && !same_call(&c->cq[0], p)) { s3_stats.pf_miss++; drain(k); }
        else if (c->ncq && k == 0) s3_stats.pf_hit++;
        if (!c->ncq) { c->cq[0] = *p; c->ncq = 1; if (k == 0) s3_stats.pf_free++; }
    }
    pump(0); pump(1);
}
// engine hint (itofs_qnext_fn): a call with these weights follows the running / next one
void s3_prefetch(const int8_t *w, int in, int out, int rows, int rb, void *user)
{
    (void)user;
    if (s3_wmode != 3 || !esp_ptr_external_ram(w)) return;
    const pcall_t p = { w, in, out, rows, rb };
    core_ctx_t *c0 = &g_cc[0];
    if (c0->ncq && same_call(&c0->cq[c0->ncq - 1], &p)) return;        // already queued (the layer's own first call after the previous layer's hint)
    if (c0->ncq >= MAXQ) return;
    for (int k = 0; k < 2; k++) g_cc[k].cq[g_cc[k].ncq++] = p;
    pump(0); pump(1);
}
// all staging idle (before the buffers are used for anything else)
void s3_stream_reset(void) { for (int k = 0; k < 2; k++) if (g_cc[k].fetched) drain(k); }

static TaskHandle_t s_worker, s_waiter;
static job_t s_job;
// work handed to core 1: a GEMM half (s_kind 0) or a part of a row-parallel engine loop (s_kind 1)
static volatile int s_kind;
static itofs_body_fn s_body; static void *s_barg; static int s_b0, s_b1;

#ifdef ITOFS_ICPROF
// QEMU -icount profiling build only: QEMU's virtual clock advances with the instructions of BOTH cores, so to get exact
// per-core counts the dual-core split is serialised (core 1's half first while core 0 sleeps in WAITI, then core 0's
// half), each half timed on its own core. Same work and same results as the parallel split; only the order differs.
#include "esp_cpu.h"
volatile int s3_ic_serial = 0;
s3_ic_t s3_ic;
static volatile uint32_t s_h1;
// event trace (ICPROF_TRACE): one record per GEMM call / row-parallel loop with its exact per-core ticks and the core-0-only gap before it
s3_ev_t *s3_trace; int s3_trace_n, s3_trace_max; volatile int s3_trace_on;
static uint32_t s_tr_last;
static inline void tr_rec(int kind, uint32_t t_begin, uint32_t h0, uint32_t h1, uint32_t ovh, uint32_t bytes, int rows, int in, int out)
{
    if (s3_trace_on && s3_trace && s3_trace_n < s3_trace_max) {
        s3_ev_t *e = &s3_trace[s3_trace_n++];
        e->gap = t_begin - s_tr_last; e->h0 = h0; e->h1 = h1; e->ovh = ovh; e->bytes = bytes; e->kind = (uint16_t)kind; e->rows = (uint16_t)rows; e->in = (uint16_t)in; e->out = (uint16_t)out;
    }
    s_tr_last = esp_cpu_get_cycle_count();
}
void s3_trace_mark(void) { tr_rec(9, esp_cpu_get_cycle_count(), 0, 0, 0, 0, 0, 0, 0); }
void s3_trace_reset(void) { s3_trace_n = 0; s_tr_last = esp_cpu_get_cycle_count(); }
#endif

static void worker_task(void *arg)
{
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#ifdef ITOFS_ICPROF
        const uint32_t c0 = esp_cpu_get_cycle_count();
#endif
        if (s_kind) s_body(s_barg, s_b0, s_b1, 1);
        else gemm_range(&s_job, &g_cc[1]);
#ifdef ITOFS_ICPROF
        s_h1 = esp_cpu_get_cycle_count() - c0;
#endif
        xTaskNotifyGive(s_waiter);
    }
}

// PSRAM -> SRAM copy through the GDMA, checked (a platform where this does not work degrades to modes 0/1)
static int dma_selftest(core_ctx_t *c)
{
    const size_t n = 8192;
    int8_t *src = heap_caps_aligned_alloc(64, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!src) return 0;
    for (size_t i = 0; i < n; i++) src[i] = (int8_t)(i * 7 + 3);
    esp_cache_msync(src, n, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    memset(c->buf[0], 0, n);
    int ok = 0;
    if (esp_async_memcpy(c->dma, c->buf[0], src, n, dma_done_cb, c->sem) == ESP_OK &&
        xSemaphoreTake(c->sem, pdMS_TO_TICKS(200)) == pdTRUE) {
        ok = memcmp(c->buf[0], src, n) == 0;
        heap_caps_free(src);
    }
    return ok;               // on a timeout src is leaked on purpose: a late DMA must not write freed memory
}

// the PIE unpack must reproduce the C unpack on every (m, zp) the format allows, for arbitrary nibbles; otherwise the C code is used
static int w4_pie_check(int *ngroups)
{
    uint8_t *row = heap_caps_aligned_alloc(16, 160, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *a = heap_caps_aligned_alloc(16, 256, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT), *b = heap_caps_aligned_alloc(16, 256, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int ok = row && a && b, n = 0;
    uint32_t r = 12345u;
    for (int m = 1; m <= 7 && ok; m++)
        for (int zp = 0; zp < 16 && ok; zp++)
            for (int rep = 0; rep < 8 && ok; rep++) {          // 8 groups per row, each with this (m, zp), random nibbles (the first row of each pair all 0, then all 15)
                memset(row, 0, 160);
                for (int g = 0; g < 8; g++) {
                    for (int i = 0; i < 16; i++) { r = r * 1664525u + 1013904223u; row[g * 16 + i] = rep == 0 ? 0x00 : rep == 1 ? 0xFF : (uint8_t)(r >> 24); }
                    row[8 * 16 + g] = (uint8_t)((m << 4) | zp);
                }
                itofs_s3_w4_unpack(a, row, 8);
                w4_unpack_row(row, 256, b);
                if (memcmp(a, b, 256)) ok = 0;
                n += 8;
            }
    heap_caps_free(row); heap_caps_free(a); heap_caps_free(b);
    *ngroups = n;
    return ok;
}

void s3_kernels_init(void)
{
    g_tiles_ok = 1;
    for (int k = 0; k < 2; k++) {
        core_ctx_t *c = &g_cc[k];
        for (int i = 0; i < 2; i++) {
            c->buf[i] = heap_caps_aligned_alloc(64, TILE_BYTES + TILE_SLACK, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
            if (!c->buf[i]) g_tiles_ok = 0;
        }
        c->sem = xSemaphoreCreateCounting(4, 0);
        c->pk = heap_caps_aligned_alloc(16, PACK_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);     // packed weight row (see compute())
    }
    g_dma_ok = g_tiles_ok;
    {
        int ng = 0;
        s3_w4_pie = w4_pie_check(&ng);
        printf("kernels: int4 unpack %s (PIE kernel %s the C unpack on %d groups, every (m, zp))\n", s3_w4_pie ? "by the PIE kernel" : "by C code",
               s3_w4_pie ? "reproduces" : "FAILED to reproduce", ng);
    }
#ifdef ITOFS_EMU_DMA
    printf("kernels: PIE int8 dot (esp-nn), dual core; staging tiles %s (%d B x 2 per core in internal SRAM); GDMA EMULATED in software (QEMU)\n",
           g_tiles_ok ? "ok" : "UNAVAILABLE", TILE_BYTES);
    if (!g_tiles_ok) s3_wmode = 0;
    xTaskCreatePinnedToCore(worker_task, "itofs_w1", 4096, NULL, 10, &s_worker, 1);
    return;
#endif
    for (int k = 0; k < 2 && g_dma_ok; k++) {
        core_ctx_t *c = &g_cc[k];
        const size_t bursts[3] = { 64, 32, 16 };
        for (int i = 0; i < 3 && !c->dma; i++) {
            async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
            cfg.backlog = 4;
            cfg.dma_burst_size = bursts[i];
            if (esp_async_memcpy_install(&cfg, &c->dma) != ESP_OK) c->dma = NULL;
        }
        if (!c->dma || !dma_selftest(c)) g_dma_ok = 0;
    }
    printf("kernels: PIE int8 dot (esp-nn), dual core; staging tiles %s (%d B x 2 per core in internal SRAM); "
           "GDMA PSRAM->SRAM %s\n", g_tiles_ok ? "ok" : "UNAVAILABLE", TILE_BYTES, g_dma_ok ? "self-test ok" : "unavailable (self-test failed or no channel)");
    if (!g_tiles_ok) s3_wmode = 0;
    xTaskCreatePinnedToCore(worker_task, "itofs_w1", 4096, NULL, 10, &s_worker, 1);
}

int s3_set_wmode(int m)
{
    if (m < 0 || m > 3 || (m >= 1 && !g_tiles_ok) || (m >= 2 && !g_dma_ok)) return -1;
    if (s3_wmode >= 2 && m != s3_wmode) for (int k = 0; k < 2; k++) drain(k);       // leave the stream empty
    s3_wmode = m;
    return 0;
}

static void qgemm_common(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int rb, int w4, int out, int32_t *acc);
void s3_qgemm(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user)
{
    (void)user;
    qgemm_common(x, rows, ldx, in, w, in, 0, out, acc);
}
void s3_qgemm4(const int8_t *x, int rows, int ldx, int in, const uint8_t *w4, int rs, int out, int32_t *acc, void *user)
{
    (void)user;
    qgemm_common(x, rows, ldx, in, (const int8_t *)w4, rs, 1, out, acc);
}
static void qgemm_common(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int rb, int w4, int out, int32_t *acc)
{
    const int64_t t0 = esp_timer_get_time();
    int mode = s3_wmode;
    if (!esp_ptr_external_ram(w)) mode = 0;      // internal RAM or flash-mapped weights: in place
    job_t j = { x, rows, ldx, in, w, out, acc, 0, out, mode, rb, w4 };
    if (mode >= 2) { const pcall_t pc = { w, in, out, rows, rb }; stream_enter(&pc); }
#ifdef ITOFS_ICPROF
    const uint32_t ic0 = esp_cpu_get_cycle_count();
#endif
    const int single = !s3_dual_enabled || (long)rows * in * out < DUAL_MIN_MACS || out < 2 * TOC;
    if (single) {
        gemm_range(&j, &g_cc[0]);
        if (mode >= 2) stream_pop(1);              // core 1 had no part in this call
#ifdef ITOFS_ICPROF
        s3_ic.single += esp_cpu_get_cycle_count() - ic0; s3_ic.nsingle++;
        tr_rec(0, ic0, esp_cpu_get_cycle_count() - ic0, 0, 0, (uint32_t)rb * out, rows, in, out);
#endif
    }
#ifdef ITOFS_ICPROF
    else if (s3_ic_serial) {
        const int split = ((out / 2) + TOC - 1) / TOC * TOC;
        s_job = j; s_job.oc0 = split; s_kind = 0;
        j.oc1 = split;
        s_waiter = xTaskGetCurrentTaskHandle();
        xTaskNotifyGive(s_worker);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        const uint32_t a = esp_cpu_get_cycle_count();
        gemm_range(&j, &g_cc[0]);
        const uint32_t h0 = esp_cpu_get_cycle_count() - a, h1 = s_h1;
        s3_ic.h0 += h0; s3_ic.h1 += h1; s3_ic.hmax += h0 > h1 ? h0 : h1; s3_ic.ndual++;
        s3_ic.dual_total += esp_cpu_get_cycle_count() - ic0;
        tr_rec(1, ic0, h0, h1, esp_cpu_get_cycle_count() - ic0 - h0 - h1, (uint32_t)rb * out, rows, in, out);
    }
#endif
    else {
        const int split = ((out / 2) + TOC - 1) / TOC * TOC;
        s_job = j; s_job.oc0 = split; s_kind = 0;
        j.oc1 = split;
        s_waiter = xTaskGetCurrentTaskHandle();
        xTaskNotifyGive(s_worker);
        gemm_range(&j, &g_cc[0]);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    }
    s3_stats.us += esp_timer_get_time() - t0;
    s3_stats.macs += (double)rows * in * out;
    s3_stats.wbytes += (double)rb * out;
    s3_stats.calls++;
}

// Row-parallel engine loops (itofs_ctx_t.par): rows [n/2, n) on core 1, [0, n/2) on core 0. The engine's bodies write
// disjoint rows with per-core scratch, so the result is the same bit for bit (host test D checks every split).
void s3_par(void *user, itofs_body_fn body, void *arg, int n)
{
    (void)user;
    if (!s3_dual_enabled || n < 2) { body(arg, 0, n, 0); return; }
    const int h = n / 2;
    s_body = body; s_barg = arg; s_b0 = h; s_b1 = n; s_kind = 1;
    s_waiter = xTaskGetCurrentTaskHandle();
#ifdef ITOFS_ICPROF
    const uint32_t ic0 = esp_cpu_get_cycle_count();
    if (s3_ic_serial) {
        xTaskNotifyGive(s_worker);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        const uint32_t a = esp_cpu_get_cycle_count();
        body(arg, 0, h, 0);
        const uint32_t h0 = esp_cpu_get_cycle_count() - a, h1 = s_h1;
        s3_ic.h0 += h0; s3_ic.h1 += h1; s3_ic.hmax += h0 > h1 ? h0 : h1; s3_ic.npar++;
        s3_ic.dual_total += esp_cpu_get_cycle_count() - ic0;
        tr_rec(2, ic0, h0, h1, esp_cpu_get_cycle_count() - ic0 - h0 - h1, 0, n, 0, 0);
        return;
    }
#endif
    xTaskNotifyGive(s_worker);
    body(arg, 0, h, 0);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// ------------------------------------------------------------------------------------------------------------------
// microbenchmark: PSRAM bandwidth and the engine's GEMM shapes, every staging mode, 1 and 2 cores
// ------------------------------------------------------------------------------------------------------------------
// weights: wbase .. wbase + wbytes (the model blob in PSRAM: values are irrelevant, and its 5+ MB are far larger than
// the 64 KB cache, as in real use), cycled through block by block
static double bench_shape(const int8_t *wbase, size_t wbytes, int in, int out, int rows, int mode, int dual, int sram, int32_t *acc, const int8_t *x)
{
    const int ldx = (in + 15) & ~15;
    const size_t wsz = (size_t)in * out;
    const int nblk = sram ? 1 : (int)(wbytes / wsz);
    const int saved_mode = s3_wmode, saved_dual = s3_dual_enabled;
    s3_wmode = mode; s3_dual_enabled = dual;
    int reps = 0;
    const int64_t t0 = esp_timer_get_time();
    int64_t t1 = t0;
    double macs = 0;
    while (t1 - t0 < 300000 || reps < 4) {     // >= 0.3 s per point
        const int8_t *w = wbase + (size_t)(reps % (nblk > 0 ? nblk : 1)) * wsz;
        s3_qgemm(x, rows, ldx, in, w, out, acc, NULL);
        macs += (double)rows * in * out;
        reps++;
        t1 = esp_timer_get_time();
    }
    s3_wmode = saved_mode; s3_dual_enabled = saved_dual;
    return macs / ((double)(t1 - t0) * 1e3);   // GMAC/s
}

void s3_bench(const void *wbase_v, size_t wbytes)
{
    s3_stream_reset();
    printf("BENCH begin (QEMU numbers are meaningless; on the board send everything up to BENCH end)\n");
    const int8_t *wps = (const int8_t *)(((uintptr_t)wbase_v + 63) & ~(uintptr_t)63);
    if (wbytes > (4u << 20)) wbytes = 4u << 20;
    wbytes = wbytes > 128 ? wbytes - 128 : 0;
    int8_t *x = heap_caps_aligned_alloc(16, 16 * 1216 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *acc = heap_caps_aligned_alloc(16, 16 * 256 * 4 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!x || !acc || !g_tiles_ok) { printf("BENCH ERROR: no memory\n"); goto out; }
    for (int i = 0; i < 16 * 1216 + 64; i++) x[i] = (int8_t)(i * 40503u >> 8);
    printf("BENCH weights: %s, %u KB cycled through (data cache 64 KB)\n", esp_ptr_external_ram(wps) ? "model blob in PSRAM" :
           "model blob NOT in PSRAM (memory-mapped flash): numbers are for flash", (unsigned)(wbytes >> 10));
    {   // PSRAM -> SRAM bandwidth: memcpy and GDMA, 12 KB pieces
        int8_t *dst = g_cc[0].buf[0];
        const size_t nb = wbytes / TILE_BYTES * TILE_BYTES;
        int64_t t0 = esp_timer_get_time();
        for (size_t o = 0; o < nb; o += TILE_BYTES) memcpy(dst, wps + o, TILE_BYTES);
        int64_t dt = esp_timer_get_time() - t0;
        printf("BENCH_BW psram->sram memcpy: %.1f MB/s\n", (double)nb / (dt + 1));
        if (g_dma_ok && esp_ptr_external_ram(wps)) {
            s3_dma_errors = 0;
            t0 = esp_timer_get_time();
            for (size_t o = 0; o < nb; o += TILE_BYTES) { dma_issue(&g_cc[0], 0, wps + o, TILE_BYTES); dma_wait(&g_cc[0]); }
            dt = esp_timer_get_time() - t0;
            printf("BENCH_BW psram->sram gdma: %.1f MB/s%s\n", (double)nb / (dt + 1), s3_dma_errors ? " (DMA ERRORS)" : "");
        } else printf("BENCH_BW psram->sram gdma: unavailable\n");
        t0 = esp_timer_get_time();
        volatile int32_t sink = 0;
        for (size_t o = 0; o < nb; o += 64) sink += wps[o];          // one byte per cache line: the cache-fill rate
        dt = esp_timer_get_time() - t0;
        printf("BENCH_BW psram read through the data cache (1 byte per 64-byte line): %.1f MB/s\n", (double)nb / (dt + 1));
        (void)sink;
    }
    // the engine's GEMM shapes: (in, out per call, rows = frames x activation planes). The engine runs 24-frame chunks (the wide 1202-input layer
    // 8 frames); 16 rows is as many as the benchmark's scratch (the little free internal SRAM at boot) allows, and is enough to see the weight reuse
    static const struct { const char *name; int in, out, rows; } sh[] = {
        {"vocoder pw1 256->256(x3)", 256, 256, 16}, {"vocoder pw2 768->256", 768, 256, 16}, {"harm_proj 1202->256", 1202, 256, 8},
        {"head 256->256(x5)", 256, 256, 16}, {"mel conv 176->176(x5 taps)", 176, 176, 16},
        {"prosody conv w16a16 96->96", 96, 96, 16}, {"text conv w16a16 128->128", 128, 128, 16},
    };
    for (size_t s = 0; s < sizeof sh / sizeof sh[0]; s++) {
        printf("BENCH_GEMM %-28s rows %2d |", sh[s].name, sh[s].rows);
        for (int mode = 0; mode < 3; mode++) {
            if ((mode >= 1 && !g_tiles_ok) || (mode == 2 && !g_dma_ok)) { printf(" %s n/a |", s3_wmode_name(mode)); continue; }
            double g1 = bench_shape(wps, wbytes, sh[s].in, sh[s].out, sh[s].rows, mode, 0, 0, acc, x);
            double g2 = bench_shape(wps, wbytes, sh[s].in, sh[s].out, sh[s].rows, mode, 1, 0, acc, x);
            printf(" %s 1c %.3f 2c %.3f |", s3_wmode_name(mode), g1, g2);
        }
        printf(" GMAC/s (weights in PSRAM)\n");
    }
    {   // compute ceiling: weights resident in internal SRAM (one 12 KB staging buffer = 256 x 48)
        int8_t *wsr = g_cc[1].buf[1];
        for (int i = 0; i < TILE_BYTES; i++) wsr[i] = (int8_t)(i * 97u);
        const int ro = TILE_BYTES / 256;
        double g1 = bench_shape(wsr, TILE_BYTES, 256, ro, 8, 0, 0, 1, acc, x), g2 = bench_shape(wsr, TILE_BYTES, 256, ro, 8, 0, 1, 1, acc, x);
        printf("BENCH_GEMM 256->%d SRAM-resident rows  8 | sram 1c %.3f 2c %.3f | GMAC/s (compute ceiling, weights in internal SRAM)\n", ro, g1, g2);
    }
out:
    heap_caps_free(x); heap_caps_free(acc);
    printf("BENCH end of kernel part\n");
}
