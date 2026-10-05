// ESP32-S3 int8 GEMM for the ItoFS engine: esp-nn's PIE dot primitive, weight-stationary tiles, dual core,
// optional weight staging PSRAM -> internal SRAM (memcpy or GDMA double buffer).
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "itofs.h"
void s3_kernels_init(void);
void s3_qgemm4(const int8_t *x, int rows, int ldx, int in, const uint8_t *w4, int rs, int out, int32_t *acc, void *user);   // int4-weight rows (see itofs.h)
void s3_qgemm(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user);
extern volatile int s3_dual_enabled;
extern volatile int s3_w4_pie;            // 1: PIE unpack of int4 rows (verified against the C unpack at boot), 0: C unpack; the `unpack` command switches it
extern volatile int s3_wmode;             // 0 direct (through the cache), 1 copy (memcpy tiles), 2 gdma (double-buffered tiles), 3 gdma + cross-call prefetch
extern volatile int s3_dma_errors;
int s3_set_wmode(int m);                  // 0 or -1 if the mode is unavailable
void s3_prefetch(const int8_t *w, int in, int out, int rows, int rb, void *user);   // itofs_qnext_fn: only acts in mode 3
void s3_stream_reset(void);
int s3_dma_ok(void);
const char *s3_wmode_name(int m);
typedef struct { int64_t us; double macs, wbytes; long calls, pf_hit, pf_miss, pf_free; } s3_stats_t;   // pf_*: staged calls whose weights were already announced / announced differently / not announced
extern s3_stats_t s3_stats;               // accumulated by s3_qgemm: wall time inside GEMM calls, int8 MACs, weight bytes
void s3_stats_reset(void);
void s3_bench(const void *wbase, size_t wbytes);
void s3_par(void *user, itofs_body_fn body, void *arg, int n);   // itofs_ctx_t.par: rows split across both cores   // PSRAM bandwidth + the engine's GEMM shapes, every mode, 1 and 2 cores
#ifdef ITOFS_ICPROF
// QEMU -icount profiling (CCOUNT ticks): single = GEMM calls on core 0 alone; h0 / h1 = the two halves of the dual-core
// calls and of the row-parallel loops (s3_par; run one after the other when s3_ic_serial), hmax = sum over calls of the
// longer half; dual_total = whole calls
typedef struct { double single, h0, h1, hmax, dual_total; long nsingle, ndual, npar; } s3_ic_t;
extern s3_ic_t s3_ic;
extern volatile int s3_ic_serial;
typedef struct { uint32_t gap, h0, h1, ovh, bytes; uint16_t kind, rows, in, out; } s3_ev_t;   // kind 0 single-core GEMM, 1 dual GEMM, 2 row-parallel loop, 9 mark
extern s3_ev_t *s3_trace; extern int s3_trace_n, s3_trace_max; extern volatile int s3_trace_on;
void s3_trace_mark(void); void s3_trace_reset(void);
#endif
