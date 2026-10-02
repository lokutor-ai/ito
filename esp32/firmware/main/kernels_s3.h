// ESP32-S3 int8 GEMM for the ItoFS engine: esp-nn's PIE dot primitive, weight-stationary tiles, dual core,
// optional weight staging PSRAM -> internal SRAM (memcpy or GDMA double buffer).
#pragma once
#include <stdint.h>
#include <stddef.h>
void s3_kernels_init(void);
void s3_qgemm(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user);
extern volatile int s3_dual_enabled;
extern volatile int s3_wmode;             // 0 direct (through the cache), 1 copy (memcpy tiles), 2 gdma (double-buffered tiles)
extern volatile int s3_dma_errors;
int s3_set_wmode(int m);                  // 0 or -1 if the mode is unavailable
int s3_dma_ok(void);
const char *s3_wmode_name(int m);
typedef struct { int64_t us; double macs, wbytes; long calls; } s3_stats_t;
extern s3_stats_t s3_stats;               // accumulated by s3_qgemm: wall time inside GEMM calls, int8 MACs, weight bytes
void s3_stats_reset(void);
void s3_bench(const void *wbase, size_t wbytes);   // PSRAM bandwidth + the engine's GEMM shapes, every mode, 1 and 2 cores
