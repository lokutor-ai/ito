// Exactness check and cycle benchmark of the int8 GEMM kernels on the ESP32-S3 (build with ITOFS_KCHECK; call s3_kernel_check()).
//
// Every kernel variant must reproduce a plain C int32 reference bit for bit on random data with extreme values, for the shapes the
// engine uses and the edge cases (one vector, one row, many rows). With bench set it also measures each variant in CPU cycles per
// MAC, with the weights resident in internal SRAM (the compute ceiling) and streamed from PSRAM through the data cache (the
// engine's case), so a kernel change can be judged without running the synthesiser. Cycle numbers from QEMU mean nothing.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "kernels_s3.h"

typedef void (*gemm_fn)(const int8_t *x, int rows, int ldx, int in16, const int8_t *w, int nw, int32_t *acc, int out);

typedef struct { const char *name; gemm_fn fn; } variant_t;

static const variant_t VARIANTS[] = {
    { "rows", s3_gemm_rows },
};
#define NVARIANT ((int)(sizeof VARIANTS / sizeof VARIANTS[0]))

#ifdef ITOFS_QEMU
#define REST() ((void)0)
#else
#define REST() vTaskDelay(1)
#endif

static uint32_t s_rng = 0x2545F491u;
static uint32_t rnd(void) { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5; return s_rng; }

static void fill(int8_t *p, size_t n, int mode)
{
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = rnd();
        p[i] = mode == 0 ? (int8_t)r : mode == 1 ? (int8_t)((r & 1) ? -128 : 127) : (int8_t)-128;
    }
}

static void reference(const int8_t *x, int rows, int ldx, int in16, const int8_t *w, int nw, int32_t *acc, int out)
{
    for (int o = 0; o < nw; o++)
        for (int r = 0; r < rows; r++) {
            int32_t a = 0;
            for (int i = 0; i < in16 * 16; i++) a += (int32_t)x[(size_t)r * ldx + i] * w[(size_t)o * in16 * 16 + i];
            acc[(size_t)r * out + o] = a;
        }
}

static int check_shape(const variant_t *v, int rows, int in16, int nw, int mode, int8_t *x, int8_t *w, int32_t *got, int32_t *want)
{
    const int ldx = in16 * 16, out = nw + 3;
    fill(x, (size_t)rows * ldx, mode);
    fill(w, (size_t)nw * in16 * 16, mode == 2 ? 2 : (mode + 1) % 3);
    for (int i = 0; i < rows * out; i++) got[i] = want[i] = 0x5A5A5A5A;
    reference(x, rows, ldx, in16, w, nw, want, out);
    v->fn(x, rows, ldx, in16, w, nw, got, out);
    REST();
    return memcmp(got, want, (size_t)rows * out * 4) == 0;
}

static int check_all(void)
{
    static const int ROWS[] = { 1, 2, 3, 7, 8, 12, 16, 20, 24 };
    static const int IN16[] = { 1, 2, 3, 6, 8, 11, 16, 48, 76 };
    static const int NW[] = { 1, 5, 16, 33 };
    int8_t *x = heap_caps_aligned_alloc(16, 24 * 76 * 16 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *w = heap_caps_aligned_alloc(64, 33 * 76 * 16 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *got = heap_caps_aligned_alloc(16, 24 * 40 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *want = heap_caps_aligned_alloc(16, 24 * 40 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!x || !w || !got || !want) { printf("KCHECK ERROR no memory\n"); return 0; }
    int ok_all = 1;
    for (int v = 0; v < NVARIANT; v++) {
        int bad = 0, n = 0;
        for (size_t a = 0; a < sizeof ROWS / sizeof ROWS[0]; a++)
            for (size_t b = 0; b < sizeof IN16 / sizeof IN16[0]; b++)
                for (size_t c = 0; c < sizeof NW / sizeof NW[0]; c++)
                    for (int mode = 0; mode < 3; mode++) {
                        n++;
                        if (!check_shape(&VARIANTS[v], ROWS[a], IN16[b], NW[c], mode, x, w, got, want)) {
                            if (bad++ < 4) printf("KCHECK %s DIFFERENT rows=%d in16=%d nw=%d mode=%d\n", VARIANTS[v].name, ROWS[a], IN16[b], NW[c], mode);
                        }
                    }
        printf("KCHECK %s %d/%d %s\n", VARIANTS[v].name, n - bad, n, bad ? "DIFFERENT" : "IDENTICAL");
        if (bad) ok_all = 0;
    }
    heap_caps_free(x); heap_caps_free(w); heap_caps_free(got); heap_caps_free(want);
    return ok_all;
}

static double bench_one(const variant_t *v, int rows, int in16, int nw, const int8_t *x, const int8_t *wbase, size_t wspan, int32_t *acc)
{
    const size_t wsz = (size_t)nw * in16 * 16;
    const size_t nblk = wspan / wsz ? wspan / wsz : 1;
    const int ldx = in16 * 16;
    uint32_t cycles = 0;
    long calls = 0;
    for (int rep = 0; rep < 16; rep++) {
        const int8_t *w = wbase + (size_t)(rep % nblk) * wsz;
        const uint32_t t0 = esp_cpu_get_cycle_count();
        v->fn(x, rows, ldx, in16, w, nw, acc, nw);
        cycles += esp_cpu_get_cycle_count() - t0;
        calls++;
    }
    REST();
    return (double)cycles / ((double)calls * rows * nw * in16 * 16);
}

static void bench_all(void)
{
    static const struct { int rows, in16, nw; } SH[] = {
        { 8, 1, 64 }, { 8, 2, 64 }, { 8, 4, 64 }, { 8, 8, 64 }, { 8, 16, 64 }, { 8, 48, 32 },
        { 20, 16, 64 }, { 20, 6, 64 }, { 20, 11, 64 }, { 20, 48, 32 }, { 4, 16, 64 },
    };
    const size_t WS_SRAM = 32 * 1024, WS_PSRAM = 4096 * 1024;
    int8_t *x = heap_caps_aligned_alloc(16, 24 * 76 * 16 + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int32_t *acc = heap_caps_aligned_alloc(16, 24 * 80 * 4, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *ws = heap_caps_aligned_alloc(64, WS_SRAM, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *wp = heap_caps_aligned_alloc(64, WS_PSRAM, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!x || !acc || !ws || !wp) { printf("KBENCH ERROR no memory (x %p acc %p ws %p wp %p)\n", (void *)x, (void *)acc, (void *)ws, (void *)wp); return; }
    fill(x, 24 * 76 * 16, 0); fill(ws, WS_SRAM, 0); fill(wp, WS_PSRAM, 0);
    const size_t win = (WS_PSRAM / NVARIANT) & ~(size_t)63;
    for (size_t s = 0; s < sizeof SH / sizeof SH[0]; s++) {
        printf("KBENCH rows=%2d in=%4d out=%2d |", SH[s].rows, SH[s].in16 * 16, SH[s].nw);
        for (int v = 0; v < NVARIANT; v++) {
            const double sr = bench_one(&VARIANTS[v], SH[s].rows, SH[s].in16, SH[s].nw, x, ws, WS_SRAM, acc);
            const double ps = bench_one(&VARIANTS[v], SH[s].rows, SH[s].in16, SH[s].nw, x, wp + (size_t)v * win, win, acc);
            printf(" %s sram %.3f psram %.3f |", VARIANTS[v].name, sr, ps);
        }
        printf(" cycles/MAC\n");
    }
    heap_caps_free(x); heap_caps_free(acc); heap_caps_free(ws); heap_caps_free(wp);
}

int s3_kernel_check(int bench)
{
    if (bench) bench_all();
    const int ok = check_all();
    printf("KCHECK %s\n", ok ? "ALL IDENTICAL" : "SOME DIFFERENT");
    return ok;
}
