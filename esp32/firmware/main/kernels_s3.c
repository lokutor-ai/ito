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
//              boot self-test of the async-memcpy channel (QEMU does not emulate it -> mode unavailable there).
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
#define TILE_BYTES 12288        // per staging buffer (2 per core): 48 KB of internal SRAM in total
#define TILE_SLACK 128          // GDMA moves 64-byte aligned supersets of the tile

volatile int s3_dual_enabled = 1;
volatile int s3_wmode = 0;
volatile int s3_dma_errors = 0;
s3_stats_t s3_stats;

typedef struct { int8_t *buf[2]; async_memcpy_handle_t dma; SemaphoreHandle_t sem; } core_ctx_t;
static core_ctx_t g_cc[2];
static int g_dma_ok, g_tiles_ok;

typedef struct { const int8_t *x; int rows, ldx, in; const int8_t *w; int out; int32_t *acc; int oc0, oc1; int mode; } job_t;

const char *s3_wmode_name(int m) { return m == 0 ? "direct" : m == 1 ? "copy" : "gdma"; }
int s3_dma_ok(void) { return g_dma_ok; }
void s3_stats_reset(void) { memset(&s3_stats, 0, sizeof s3_stats); }

// main/dot_rows_s3.S: one aligned weight row against every activation row (2 instructions per 16 MACs)
extern void itofs_s3_dot_rows(const int8_t *x, int ldx, const int8_t *w, int rows_len16, int32_t *acc, int stride_bytes);

// channels [o0, o1) whose weights start at wt (row stride in)
static inline void compute(const job_t *j, const int8_t *wt, int o0, int o1)
{
    const int in = j->in, in16 = in >> 4, r0 = in16 << 4;
    if ((in & 15) == 0 && ((uintptr_t)wt & 15) == 0 && j->rows < 65536) {   // every weight row aligned: fast kernel
        for (int o = o0; o < o1; o++)
            itofs_s3_dot_rows(j->x, j->ldx, wt + (size_t)(o - o0) * in, (j->rows << 16) | in16, j->acc + o, j->out * 4);
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

// start copying the 64-byte aligned superset of [src, src + n) into buffer bi; returns where the tile starts in it
static const int8_t *dma_issue(core_ctx_t *c, int bi, const int8_t *src, size_t n)
{
    const uintptr_t a = (uintptr_t)src & ~(uintptr_t)63, e = ((uintptr_t)src + n + 63) & ~(uintptr_t)63;
    if (esp_async_memcpy(c->dma, c->buf[bi], (void *)a, e - a, dma_done_cb, c->sem) != ESP_OK) {
        s3_dma_errors++;
        memcpy(c->buf[bi], (const void *)a, e - a);
        xSemaphoreGive(c->sem);
    }
    return c->buf[bi] + ((uintptr_t)src - a);
}

static void dma_wait(core_ctx_t *c)
{
    if (xSemaphoreTake(c->sem, pdMS_TO_TICKS(500)) != pdTRUE) s3_dma_errors++;
}

static void gemm_range(const job_t *j, core_ctx_t *c)
{
    if (j->oc1 <= j->oc0) return;
    const int in = j->in;
    if (j->mode == 0) { compute(j, j->w + (size_t)j->oc0 * in, j->oc0, j->oc1); return; }
    int toc = (TILE_BYTES / in) & ~(TOC - 1);
    if (toc < TOC) toc = TOC;                     // in <= 1536 always (the engine's widest GEMM input is 1202)
    if (j->mode == 1) {
        for (int t0 = j->oc0; t0 < j->oc1; t0 += toc) {
            const int t1 = (t0 + toc < j->oc1) ? t0 + toc : j->oc1;
            memcpy(c->buf[0], j->w + (size_t)t0 * in, (size_t)(t1 - t0) * in);
            compute(j, c->buf[0], t0, t1);
        }
        return;
    }
    const int nt = (j->oc1 - j->oc0 + toc - 1) / toc;
    int t1 = (j->oc0 + toc < j->oc1) ? j->oc0 + toc : j->oc1;
    const int8_t *cur = dma_issue(c, 0, j->w + (size_t)j->oc0 * in, (size_t)(t1 - j->oc0) * in);
    for (int i = 0; i < nt; i++) {
        const int t0 = j->oc0 + i * toc;
        t1 = (t0 + toc < j->oc1) ? t0 + toc : j->oc1;
        dma_wait(c);
        const int8_t *nxt = NULL;
        if (i + 1 < nt) {
            const int n1 = (t1 + toc < j->oc1) ? t1 + toc : j->oc1;
            nxt = dma_issue(c, (i + 1) & 1, j->w + (size_t)t1 * in, (size_t)(n1 - t1) * in);
        }
        compute(j, cur, t0, t1);
        cur = nxt;
    }
}

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

void s3_kernels_init(void)
{
    g_tiles_ok = 1;
    for (int k = 0; k < 2; k++) {
        core_ctx_t *c = &g_cc[k];
        for (int i = 0; i < 2; i++) {
            c->buf[i] = heap_caps_aligned_alloc(64, TILE_BYTES + TILE_SLACK, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
            if (!c->buf[i]) g_tiles_ok = 0;
        }
        c->sem = xSemaphoreCreateBinary();
    }
    g_dma_ok = g_tiles_ok;
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
    if (m < 0 || m > 2 || (m >= 1 && !g_tiles_ok) || (m == 2 && !g_dma_ok)) return -1;
    s3_wmode = m;
    return 0;
}

void s3_qgemm(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user)
{
    (void)user;
    const int64_t t0 = esp_timer_get_time();
    int mode = s3_wmode;
    if (!esp_ptr_external_ram(w)) mode = 0;      // internal RAM or flash-mapped weights: in place
    job_t j = { x, rows, ldx, in, w, out, acc, 0, out, mode };
#ifdef ITOFS_ICPROF
    const uint32_t ic0 = esp_cpu_get_cycle_count();
#endif
    if (!s3_dual_enabled || (long)rows * in * out < DUAL_MIN_MACS || out < 2 * TOC) {
        gemm_range(&j, &g_cc[0]);
#ifdef ITOFS_ICPROF
        s3_ic.single += esp_cpu_get_cycle_count() - ic0; s3_ic.nsingle++;
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
    s3_stats.wbytes += (double)in * out;
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
    // the engine's GEMM calls at chunk 8: (in, out per call, rows = frames x activation planes)
    static const struct { const char *name; int in, out, rows; } sh[] = {
        {"vocoder pw1 256->256(x3)", 256, 256, 8}, {"vocoder pw2 768->256", 768, 256, 8}, {"harm_proj 1202->256", 1202, 256, 8},
        {"head 256->256(x5)", 256, 256, 8}, {"mel conv 176->176(x5 taps)", 176, 176, 8},
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
        double g1 = bench_shape(wsr, TILE_BYTES, 256, 48, 8, 0, 0, 1, acc, x), g2 = bench_shape(wsr, TILE_BYTES, 256, 48, 8, 0, 1, 1, acc, x);
        printf("BENCH_GEMM %-28s rows  8 | sram 1c %.3f 2c %.3f | GMAC/s (compute ceiling, weights in internal SRAM)\n", "256->48 SRAM-resident", g1, g2);
    }
out:
    heap_caps_free(x); heap_caps_free(acc);
    printf("BENCH end of kernel part\n");
}
