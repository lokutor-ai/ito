// Synthetic two-core lock-step stress with Ito's own PIE dot kernel (itofs_s3_dot_rows, unchanged copy).
// Shared activation tile in internal SRAM (read by BOTH cores at once), random int8 weights streamed from PSRAM through the data cache,
// outputs split half/half between the cores; every call's int32 result is hashed and compared with a one-core reference.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "esp_system.h"
#include "esp_cpu.h"
extern void itofs_s3_dot_rows(const int8_t *x, int ldx, const int8_t *w, int rows_len16, int32_t *acc, int stride_bytes);

typedef struct { const char *name; int K, rows, out, nblk, secs, dual; } cfg_t;
// out = 176 outputs per call split 88/88 (as Oido's failing case); Ito-like shapes follow
static const cfg_t CFG[] = {
    { "K704_rows64_out176_dual (Oido pattern, 45KB tile)", 704, 64, 176, 6, 90, 1 },
    { "K576_rows64_out176_dual (36KB tile)",               576, 64, 176, 6, 90, 1 },
    { "K576_rows24_out256_dual (Ito pw2-like)",            576, 24, 256, 6, 60, 1 },
    { "K192_rows24_out576_dual (Ito pw1-like)",            192, 24, 576, 6, 60, 1 },
    { "K1216_rows12_out256_dual (Ito wide-like)",         1216, 12, 256, 6, 60, 1 },
    { "K704_rows64_out176_SINGLE (control, core0 only)",   704, 64, 176, 6, 60, 0 },
};
#define NCFG (sizeof CFG / sizeof CFG[0])

static volatile uint32_t g_go, g_done;
typedef struct { const int8_t *x; int ldx, K, rows, out, o0, o1; const int8_t *w; int32_t *acc; } job_t;
static job_t g_job;
static RTC_NOINIT_ATTR uint32_t rtc_boots, rtc_magic;

static inline void half(const job_t *j, int o0, int o1)
{
    for (int o = o0; o < o1; o++)
        itofs_s3_dot_rows(j->x, j->ldx, j->w + (size_t)o * j->K, (j->rows << 16) | (j->K >> 4), j->acc + o, j->out * 4);
}
static void IRAM_ATTR worker(void *arg)
{
    uint32_t seen = 0;
    for (;;) {
        while (g_go == seen) { __asm__ volatile("nop"); }
        seen = g_go;
        half(&g_job, g_job.o1, g_job.out);       // second half
        __asm__ volatile("memw");
        g_done = seen;
    }
}
static uint32_t fnv(const int32_t *a, size_t n) { uint32_t h = 2166136261u; for (size_t i = 0; i < n; i++) { h ^= (uint32_t)a[i]; h *= 16777619u; } return h; }
static uint32_t lcg_s;
static inline uint32_t lcg(void) { lcg_s = lcg_s * 1664525u + 1013904223u; return lcg_s >> 8; }

static void run_cfg(const cfg_t *c, int round)
{
    const int K = c->K, R = c->rows, O = c->out;
    int8_t *x = heap_caps_aligned_alloc(16, (size_t)R * K, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *acc = heap_caps_aligned_alloc(16, (size_t)R * O * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *w = heap_caps_aligned_alloc(64, (size_t)c->nblk * O * K, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int32_t *ref = heap_caps_aligned_alloc(16, (size_t)c->nblk * R * O * 4, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint32_t *refh = malloc(c->nblk * 4);
    if (!x || !acc || !w || !ref) { printf("STRESS alloc failed (%p %p %p %p)\n", x, acc, w, ref); return; }
    lcg_s = 12345u + round;
    for (size_t i = 0; i < (size_t)R * K; i++) x[i] = (int8_t)lcg();
    for (size_t i = 0; i < (size_t)c->nblk * O * K; i++) w[i] = (int8_t)lcg();
    // reference: one core, twice, plus a plain C check of block 0
    int bad_ref = 0;
    for (int b = 0; b < c->nblk; b++) {
        g_job = (job_t){ x, K, K, R, O, 0, O, w + (size_t)b * O * K, acc };
        half(&g_job, 0, O);
        memcpy(ref + (size_t)b * R * O, acc, (size_t)R * O * 4);
        uint32_t h1 = fnv(acc, (size_t)R * O);
        memset(acc, 0, (size_t)R * O * 4);
        half(&g_job, 0, O);
        if (fnv(acc, (size_t)R * O) != h1) bad_ref++;
        refh[b] = h1;
    }
    if (round == 0) {      // C check of the first block's first rows
        int cbad = 0;
        for (int r = 0; r < (R < 4 ? R : 4); r++) for (int o = 0; o < O; o += 7) {
            int32_t a = 0; const int8_t *wr = w + (size_t)o * K, *xr = x + (size_t)r * K;
            for (int i = 0; i < K; i++) a += (int32_t)xr[i] * wr[i];
            if (a != ref[(size_t)r * O + o]) cbad++;
        }
        printf("STRESS C-reference check of the PIE kernel: %d mismatches\n", cbad);
    }
    printf("STRESS cfg start: %s | x tile %d B internal SRAM, weights %d KB PSRAM, one-core reference unstable blocks: %d\n", c->name, R * K, c->nblk * O * K / 1024, bad_ref);
    const int64_t t0 = esp_timer_get_time(), tend = t0 + (int64_t)c->secs * 1000000;
    long calls = 0, mism = 0, mism_persist = 0, mism_vs_single = 0; int64_t tlast = t0; int b = 0;
    while (esp_timer_get_time() < tend) {
        for (int rep = 0; rep < 50; rep++) {
            const int8_t *wb = w + (size_t)b * O * K;
            int split = ((O / 2) + 7) / 8 * 8;
            g_job = (job_t){ x, K, K, R, O, 0, c->dual ? split : O, wb, acc };
            if (c->dual) {
                uint32_t id = g_go + 1; __asm__ volatile("memw"); g_go = id;
                half(&g_job, 0, split);
                while (g_done != id) { __asm__ volatile("nop"); }
            } else half(&g_job, 0, O);
            calls++;
            if (fnv(acc, (size_t)R * O) != refh[b]) {
                mism++;
                int first = -1, nd = 0; const int32_t *rb = ref + (size_t)b * R * O;
                for (int i = 0; i < R * O; i++) if (acc[i] != rb[i]) { if (first < 0) first = i; nd++; }
                if (mism <= 20) {
                    printf("STRESS MISMATCH cfg '%s' call %ld block %d: %d of %d words differ, first at row %d out %d (got %08x want %08x)\n", c->name, calls, b, nd, R * O,
                           first / O, first % O, (unsigned)acc[first], (unsigned)rb[first]);
                    // rerun immediately, dual again
                    memset(acc, 0, (size_t)R * O * 4);
                    uint32_t id = g_go + 1; g_job = (job_t){ x, K, K, R, O, 0, split, wb, acc }; __asm__ volatile("memw"); g_go = id;
                    half(&g_job, 0, split); while (g_done != id) { __asm__ volatile("nop"); }
                    int again = fnv(acc, (size_t)R * O) != refh[b];
                    memset(acc, 0, (size_t)R * O * 4);
                    g_job.o1 = O; half(&g_job, 0, O);
                    int single_ok = fnv(acc, (size_t)R * O) == refh[b];
                    printf("STRESS   repeat dual same inputs: %s; one-core rerun: %s\n", again ? "MISMATCH AGAIN" : "ok", single_ok ? "matches reference" : "ALSO WRONG");
                    if (again) mism_persist++;
                }
            }
            b = (b + 1) % c->nblk;
        }
        if (esp_timer_get_time() - tlast > 15000000) { tlast = esp_timer_get_time(); printf("STRESS   ... %s: %ld calls, %ld mismatches\n", c->name, calls, mism); }
    }
    const double s = (esp_timer_get_time() - t0) / 1e6;
    printf("STRESS cfg done: %s | %.0f s, %ld calls (%.0f MMAC/s), mismatches %ld (repeat-persistent %ld)\n", c->name, s, calls, (double)calls * R * K * O / s / 1e6, mism, mism_persist);
    heap_caps_free(x); heap_caps_free(acc); heap_caps_free(w); heap_caps_free(ref); free(refh);
}

void app_main(void)
{
    if (rtc_magic != 0xB0A12D5Cu) { rtc_magic = 0xB0A12D5Cu; rtc_boots = 0; }
    rtc_boots++;
    printf("\nSTRESS boot #%u (reset reason %d), CPU %d MHz, internal free %u KB\n", (unsigned)rtc_boots, (int)esp_reset_reason(),
           (int)(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    xTaskCreatePinnedToCore(worker, "w1", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    vTaskDelay(10);
    for (int round = 0;; round++) {
        for (unsigned i = 0; i < NCFG; i++) run_cfg(&CFG[i], round);
        printf("STRESS round %d complete\n", round);
    }
}
