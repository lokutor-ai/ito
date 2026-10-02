// ItoFS portable C99 inference engine. See itofs.h for the numerics and the streaming design.
#include "itofs.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define ITF_MAGIC 0x31465449u   // "ITF1"
#define QMAX16 16256.0f         // 127 * 128: q splits exactly into hi, lo in [-127, 127] x [-64, 63]
#define PLANES 2                // max activation planes (sizing)
#define QMAX8 127.0f
#define ACC_TILE 256            // output channels per GEMM call (bounds the int32 accumulator scratch)
#define TEXT_BATCH 32           // tokens per GEMM batch on the whole-sentence text side: each batch re-reads the text side's
                                // int16 weights (0.9 MB) from PSRAM, so 32-token batches read them 4x less often than 8.
                                // The output-channel tile shrinks to fit the same accumulator scratch: bit-identical results.
#define FIRST_TEXT_STEP 8       // tokens the text side adds per step before the first chunk (0 = chunk_frames after it)
#define MAX_FACTORS 32
#define FILL_ROWS 2             // extra rows in the frame-batch scratch, so a 2-frame first chunk's pipeline fill runs every
                                // stage in one pass over its weights (~16 KB of internal SRAM)
#define MAX_FILL 64             // most rows a stage computes in one run (first-chunk pipeline fill; scratch permitting)
#ifndef WIDE_ROWS
#define WIDE_ROWS 8             // rows the two wide (n_fft + 2 = 1202 column) layers, harm_proj and the output head, process per weight
                                // pass: their float rows are 4.8 KB each, so they are batched less than the rest of the decoder
#endif
#define ACC_BYTES (24 * 1024)   // int32 accumulator scratch of one GEMM call; the output-channel tile shrinks to fit (bit-identical)
#define TSENT (1 << 20)         // frames: T sentinel while the incremental text side has not finished (T must stay below)

enum { E_OK = 0, E_BLOB = -1, E_MISSING = -2, E_SHAPE = -3, E_CONFIG = -4, E_ARENA = -5, E_ARG = -6, E_TOO_LONG = -7 };

const char *itofs_strerror(int err)
{
    switch (err) {
    case E_OK: return "ok";
    case E_BLOB: return "bad weight blob";
    case E_MISSING: return "tensor or key missing from blob";
    case E_SHAPE: return "tensor shape mismatch";
    case E_CONFIG: return "unsupported config";
    case E_ARENA: return "arena too small";
    case E_ARG: return "bad argument";
    case E_TOO_LONG: return "token sequence longer than max_tokens";
    default: return "error";
    }
}

static int imin(int a, int b) { return a < b ? a : b; }
static inline int pad16(int n) { return (n + 15) & ~15; }
static int imax(int a, int b) { return a > b ? a : b; }
static int floordiv(int a, int b) { int q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; }

// run body over rows [0, n) on both cores when the caller gave a par hook and the work is large enough to pay for the
// cross-core hand-off (work = rough element count); bodies write disjoint rows, so the split never changes a bit
#define PAR_MIN_WORK 1536
static void par_for(const itofs_ctx_t *c, itofs_body_fn body, void *arg, int n, long work)
{
    if (c->par && !c->tap && n > 1 && work >= PAR_MIN_WORK) c->par(c->par_user, body, arg, n);
    else if (n > 0) body(arg, 0, n, 0);
}

// ======================================================================================================================
// deterministic math: only IEEE + - * / (and sqrt / floor, which are exact), so host and chip agree bit for bit
// ======================================================================================================================
#ifdef ITOFS_OPCOUNT
long long itofs_opc[ITOFS_OPC_N];
#define OPC(k, n) (itofs_opc[k] += (n))
#else
#define OPC(k, n) ((void)0)
#endif
#ifdef ITOFS_PROF      // profiling build: time per stage kind / qlin part with a caller-supplied clock (itofs.h)
uint32_t (*itofs_prof_clock)(void);
double itofs_prof[ITOFS_PROF_N];
#define PROF_T(v) const uint32_t v = itofs_prof_clock()
#define PROF_ADD(k, v) (itofs_prof[k] += (double)(uint32_t)(itofs_prof_clock() - (v)))
#else
#define PROF_T(v) ((void)0)
#define PROF_ADD(k, v) ((void)0)
#endif

// bit casts through a union (ESP-IDF builds with -fno-builtin-memcpy, so a 4-byte memcpy would be a real call)
typedef union { float f; uint32_t u; } fbits_t;
static inline float f_from_bits(uint32_t u) { fbits_t b; b.u = u; return b.f; }
static inline uint32_t bits_of(float f) { fbits_t b; b.f = f; return b.u; }
// floor(v) as an int for |v| < 2^31: the Xtensa FPU has it as one instruction (FLOOR.S); elsewhere the same result in C
#if defined(__XTENSA__)
static inline int floor_i(float v) { int r; __asm__("floor.s %0, %1, 0" : "=a"(r) : "f"(v)); return r; }
#else
static inline int floor_i(float v) { const int i = (int)v; return (float)i > v ? i - 1 : i; }
#endif
// floorf without a libm call, identical result for every input (|v| >= 2^23, inf, NaN: already integral / unchanged)
static inline float itf_floorf(float v)
{
    if (!(fabsf(v) < 8388608.f) || v == 0.f) return v;
    return (float)floor_i(v);
}

// Constants of the hot float kernels in one table that the compiler cannot fold: GCC's Xtensa code loads a float literal with two
// instructions (l32r + wfr), but a table read through a pointer is one (lsi). The values are the literals the plain versions use.
enum { KT_ERF_C = 0, KT_SQRT_HALF = 13, KT_2_SQRTPI = 14, KT_ERF_A = 15 /* 0.3275911 */, KT_ERF_P1 = 16, KT_ERF_P2, KT_ERF_P3, KT_ERF_P4, KT_ERF_P5,
       KT_LOG2E = 21, KT_HALF = 22, KT_LN2_HI = 23, KT_LN2_LO = 24, KT_E2 = 25 /* 1/2 */, KT_E3 /* 1/6 */, KT_E4 /* 1/24 */, KT_E5, KT_E6, KT_E7,
       KT_EXP_HI = 31, KT_EXP_LO = 32, KT_S_2PI = 33, KT_S_P1 = 34, KT_S_P2 = 35, KT_S_P3 = 36, KT_S1 = 37 /* -1/6 */, KT_S2, KT_S3, KT_S4, KT_S5,
       KT_C1 = 42 /* -0.5 */, KT_C2, KT_C3, KT_C4, KT_C5, KT_C6, KT_N = 48 };
static const float ITF_KT[KT_N] = {
    1.0f, -0.333333333333333333f, 0.1f, -0.0238095238095238095f, 0.00462962962962962963f, -0.000757575757575757576f,
    0.000106837606837606838f, -1.32275132275132275e-05f, 1.45038522231464454e-06f, -1.42308235157637989e-07f,
    1.25822028025231796e-08f, -1.01070949532390185e-09f, 7.42528949736183659e-11f,
    0.70710678118654752f, 1.12837916709551257f, 0.3275911f, 0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f, 1.061405429f,
    1.44269504088896341f, 0.5f, 0.693145751953125f, 1.428606765330187e-06f, 0.5f, 1.f / 6, 1.f / 24, 1.f / 120, 1.f / 720, 1.f / 5040, 88.7f, -87.3f,
    0.636619772367581343f, 1.5703125f, 4.837512969970703125e-4f, 7.54978995489188216e-8f,        /* 33..36 */
    -1.f / 6, 1.f / 120, -1.f / 5040, 1.f / 362880, -1.f / 39916800,                               /* 37..41 */
    -0.5f, 1.f / 24, -1.f / 720, 1.f / 40320, -1.f / 3628800, 1.f / 479001600 };                   /* 42..47 */
const float *itf_kt = ITF_KT;       // (non-const pointer on purpose: see above)

static inline __attribute__((always_inline)) float itf_expf_k(float x, const float *K)
{
    OPC(ITOFS_OPC_EXP, 1);
    if (x != x) return x;
    if (x > K[KT_EXP_HI]) return f_from_bits(0x7f800000u);
    if (x < K[KT_EXP_LO]) return 0.f;
    float kf = itf_floorf(x * K[KT_LOG2E] + K[KT_HALF]);
    float r = (x - kf * K[KT_LN2_HI]) - kf * K[KT_LN2_LO];
    float p = 1.f + r * (1.f + r * (K[KT_HALF] + r * (K[KT_E3] + r * (K[KT_E4] + r * (K[KT_E5] + r * (K[KT_E6] + r * K[KT_E7]))))));
    int k = (int)kf;
    if (k < -125) { p *= f_from_bits(0x00800000u) * 2.f; k += 125; p *= f_from_bits((uint32_t)(k + 127) << 23); return p; }
    return p * f_from_bits((uint32_t)(k + 127) << 23);
}
static float itf_expf(float x) { return itf_expf_k(x, itf_kt); }

static float itf_expm1f(float x)
{
    if (x > -0.35f && x < 0.35f)
        return x * (1.f + x * (0.5f + x * (1.f / 6 + x * (1.f / 24 + x * (1.f / 120 + x * (1.f / 720 + x * (1.f / 5040 + x * (1.f / 40320))))))));
    return itf_expf(x) - 1.f;
}

static float itf_tanhf(float x)
{
    float a = fabsf(x);
    if (a > 9.f) return x > 0 ? 1.f : -1.f;
    float e = itf_expm1f(2.f * a);
    float t = e / (e + 2.f);
    return x < 0 ? -t : t;
}

static float itf_sigmoidf(float x) { return 1.f / (1.f + itf_expf(-x)); }

static float itf_logf(float x)
{
    OPC(ITOFS_OPC_LOG, 1);
    if (!(x > 0.f)) return x == 0.f ? -f_from_bits(0x7f800000u) : f_from_bits(0x7fc00000u);
    uint32_t u = bits_of(x);
    int e = (int)((u >> 23) & 0xff) - 127;
    if (e == -127) return -87.3f;
    float m = f_from_bits((u & 0x007fffffu) | 0x3f800000u);
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }
    float s = (m - 1.f) / (m + 1.f), s2 = s * s;
    float l = 2.f * s * (1.f + s2 * (1.f / 3 + s2 * (1.f / 5 + s2 * (1.f / 7 + s2 * (1.f / 9)))));
    const float ln2_hi = 0.693145751953125f, ln2_lo = 1.428606765330187e-06f;
    return (float)e * ln2_hi + (l + (float)e * ln2_lo);
}

static float itf_erff(float x)
{
    OPC(ITOFS_OPC_ERF, 1);
    float a = fabsf(x), r;
    if (a <= 1.f) {
        OPC(ITOFS_OPC_ERF_SERIES, 1);
        float x2 = a * a, term = a, sum = a;
        for (int n = 1; n <= 12; n++) { term *= -x2 / (float)n; sum += term / (float)(2 * n + 1); }
        r = sum * 1.12837916709551257f;
    } else if (a < 4.f) {
        float t = 1.f / (1.f + 0.3275911f * a);
        float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        r = 1.f - p * itf_expf(-a * a);
    } else {
        r = 1.f;
    }
    return x < 0 ? -r : r;
}

// erf for arch-3 blobs: the same Maclaurin series (13 terms, |x| <= 1) evaluated in Horner form with precomputed float
// coefficients c_n = (-1)^n / (n! (2n+1)) -- no divisions (the series form above costs 24 FPU divisions per call, and
// the vocoder evaluates ~4,800 GELUs per frame). Same tail branch. arch-2 blobs keep itf_erff (their self-test hashes).
static const float ERF_C[13] = {
    1.0f, -0.333333333333333333f, 0.1f, -0.0238095238095238095f, 0.00462962962962962963f, -0.000757575757575757576f,
    0.000106837606837606838f, -1.32275132275132275e-05f, 1.45038522231464454e-06f, -1.42308235157637989e-07f,
    1.25822028025231796e-08f, -1.01070949532390185e-09f, 7.42528949736183659e-11f };

static float __attribute__((noinline)) itf_erf_tail(float a)       // 1 < a: the erf_h / erf tail branch
{
    if (a < 4.f) {
        float t = 1.f / (1.f + 0.3275911f * a);
        float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        return 1.f - p * itf_expf(-a * a);
    }
    return 1.f;
}

// (the Horner loop written out: the same operations in the same order, so the same bits, in fewer instructions)
static inline __attribute__((always_inline)) float itf_erff_h(float x)
{
    OPC(ITOFS_OPC_ERF, 1);
    float a = fabsf(x), r;
    if (a <= 1.f) {
        OPC(ITOFS_OPC_ERF_SERIES, 1);
        const float x2 = a * a;
        float p = ERF_C[12];
        p = p * x2 + ERF_C[11]; p = p * x2 + ERF_C[10]; p = p * x2 + ERF_C[9]; p = p * x2 + ERF_C[8];
        p = p * x2 + ERF_C[7]; p = p * x2 + ERF_C[6]; p = p * x2 + ERF_C[5]; p = p * x2 + ERF_C[4];
        p = p * x2 + ERF_C[3]; p = p * x2 + ERF_C[2]; p = p * x2 + ERF_C[1]; p = p * x2 + ERF_C[0];
        r = a * p * 1.12837916709551257f;
    } else {
        r = itf_erf_tail(a);
    }
    return x < 0 ? -r : r;
}

// exp(x) for -87 < x < 0 -- itf_expf without the range tests and the k < -125 branch (same operations, same bits there)
static inline __attribute__((always_inline)) float itf_exp_neg(float x, const float *K)
{
    const int ki = floor_i(x * K[KT_LOG2E] + K[KT_HALF]);
    const float kf = (float)ki;
    const float r = (x - kf * K[KT_LN2_HI]) - kf * K[KT_LN2_LO];
    float p = K[KT_E7];
    p = 1.f + r * (1.f + r * (K[KT_HALF] + r * (K[KT_E3] + r * (K[KT_E4] + r * (K[KT_E5] + r * (K[KT_E6] + r * p))))));
    return p * f_from_bits((uint32_t)(ki + 127) << 23);
}

// the arch-3 GELU, bit for bit: 0.5 x (1 + erf(x / sqrt 2)) with itf_erff_h's series (|.| <= 1), rational tail and saturation at 4
static inline __attribute__((always_inline)) float gelu_k(float x, const float *K)
{
    const float xs = x * K[KT_SQRT_HALF];
    const float a = fabsf(xs);
    float r;
    if (a <= 1.f) {
        const float x2 = a * a;
        float p = K[KT_ERF_C + 12];
        p = p * x2 + K[KT_ERF_C + 11]; p = p * x2 + K[KT_ERF_C + 10]; p = p * x2 + K[KT_ERF_C + 9]; p = p * x2 + K[KT_ERF_C + 8];
        p = p * x2 + K[KT_ERF_C + 7]; p = p * x2 + K[KT_ERF_C + 6]; p = p * x2 + K[KT_ERF_C + 5]; p = p * x2 + K[KT_ERF_C + 4];
        p = p * x2 + K[KT_ERF_C + 3]; p = p * x2 + K[KT_ERF_C + 2]; p = p * x2 + K[KT_ERF_C + 1]; p = p * x2 + K[KT_ERF_C + 0];
        r = a * p * K[KT_2_SQRTPI];
    } else if (a < 4.f) {
        const float t = 1.f / (1.f + K[KT_ERF_A] * a);
        const float p = t * (K[KT_ERF_P1] + t * (K[KT_ERF_P2] + t * (K[KT_ERF_P3] + t * (K[KT_ERF_P4] + t * K[KT_ERF_P5]))));
        r = 1.f - p * itf_exp_neg(-a * a, K);
    } else {
        r = 1.f;
    }
    return 0.5f * x * (1.0f + (xs < 0 ? -r : r));
}

static inline __attribute__((always_inline)) void itf_sincosf_k(float x, float *s, float *c, const float *K)
{
    OPC(ITOFS_OPC_SINCOS, 1);
    float kf = itf_floorf(x * K[KT_S_2PI] + K[KT_HALF]);
    float r = ((x - kf * K[KT_S_P1]) - kf * K[KT_S_P2]) - kf * K[KT_S_P3];
    float r2 = r * r;
    float sn = r * (1.f + r2 * (K[KT_S1] + r2 * (K[KT_S2] + r2 * (K[KT_S3] + r2 * (K[KT_S4] + r2 * K[KT_S5])))));
    float cs = 1.f + r2 * (K[KT_C1] + r2 * (K[KT_C2] + r2 * (K[KT_C3] + r2 * (K[KT_C4] + r2 * (K[KT_C5] + r2 * K[KT_C6])))));
    switch (((int)kf) & 3) {
    case 0: *s = sn; *c = cs; break;
    case 1: *s = cs; *c = -sn; break;
    case 2: *s = -sn; *c = -cs; break;
    default: *s = -cs; *c = sn; break;
    }
}
static void itf_sincosf(float x, float *s, float *c) { itf_sincosf_k(x, s, c, itf_kt); }

// sin/cos(2*pi*num/den) in double from + - * / only (tables at init: twiddles, window)
static void itf_sincos2pi_d(long num, long den, double *s, double *c)
{
    long q = num % den; if (q < 0) q += den;
    const long q4 = 4 * q;
    const int quad = (int)(q4 / den);          // quarter turns
    const long rem = q4 - (long)quad * den;    // angle within the quadrant = (pi/2) * rem / den
    const int comp = 2 * rem > den;            // use the complementary angle to stay within [0, pi/4]
    const double x = 1.5707963267948966 * ((double)(comp ? den - rem : rem) / (double)den);
    const double x2 = x * x;
    double sn = 0, cs = 0, term = x;
    for (int n = 1; n < 30; n += 2) { sn += term; term *= -x2 / (double)((n + 1) * (n + 2)); }
    term = 1.0;
    for (int n = 0; n < 30; n += 2) { cs += term; term *= -x2 / (double)((n + 1) * (n + 2)); }
    const double bs = comp ? cs : sn, bc = comp ? sn : cs;
    switch (quad & 3) {
    case 0: *s = bs; *c = bc; break;
    case 1: *s = bc; *c = -bs; break;
    case 2: *s = -bs; *c = -bc; break;
    default: *s = -bc; *c = bs; break;
    }
}

// ======================================================================================================================
// mixed-radix complex FFT (radix 4, 2, 3, 5; out of place; forward), after KISS FFT (BSD)
// ======================================================================================================================
typedef struct { float r, i; } cpx;
struct itofs_fft { int n; int fac[2 * MAX_FACTORS]; cpx *tw; cpx *tmp; };

#define C_MUL(m, a, b) do { (m).r = (a).r * (b).r - (a).i * (b).i; (m).i = (a).r * (b).i + (a).i * (b).r; } while (0)
#define C_ADD(m, a, b) do { (m).r = (a).r + (b).r; (m).i = (a).i + (b).i; } while (0)
#define C_SUB(m, a, b) do { (m).r = (a).r - (b).r; (m).i = (a).i - (b).i; } while (0)
#define C_ADDTO(m, a) do { (m).r += (a).r; (m).i += (a).i; } while (0)

static void bfly2(cpx *F, int fstride, const cpx *tw, int m)
{
    cpx *F2 = F + m;
    for (int k = 0; k < m; k++) {
        cpx t; C_MUL(t, F2[k], tw[k * fstride]);
        C_SUB(F2[k], F[k], t); C_ADDTO(F[k], t);
    }
}

static void bfly3(cpx *F, int fstride, const cpx *tw, int m)
{
    const cpx epi3 = tw[fstride * m];
    for (int k = 0; k < m; k++) {
        cpx s0, s1, s2, s3;
        C_MUL(s1, F[k + m], tw[k * fstride]); C_MUL(s2, F[k + 2 * m], tw[2 * k * fstride]);
        C_ADD(s3, s1, s2); C_SUB(s0, s1, s2);
        F[k + m].r = F[k].r - s3.r * 0.5f; F[k + m].i = F[k].i - s3.i * 0.5f;
        s0.r *= epi3.i; s0.i *= epi3.i;
        C_ADDTO(F[k], s3);
        F[k + 2 * m].r = F[k + m].r + s0.i; F[k + 2 * m].i = F[k + m].i - s0.r;
        F[k + m].r -= s0.i; F[k + m].i += s0.r;
    }
}

static void bfly4(cpx *F, int fstride, const cpx *tw, int m)
{
    for (int k = 0; k < m; k++) {
        cpx s0, s1, s2, s3, s4, s5;
        C_MUL(s0, F[k + m], tw[k * fstride]); C_MUL(s1, F[k + 2 * m], tw[2 * k * fstride]); C_MUL(s2, F[k + 3 * m], tw[3 * k * fstride]);
        C_SUB(s5, F[k], s1); C_ADDTO(F[k], s1);
        C_ADD(s3, s0, s2); C_SUB(s4, s0, s2);
        C_SUB(F[k + 2 * m], F[k], s3); C_ADDTO(F[k], s3);
        F[k + m].r = s5.r + s4.i; F[k + m].i = s5.i - s4.r;
        F[k + 3 * m].r = s5.r - s4.i; F[k + 3 * m].i = s5.i + s4.r;
    }
}

static void bfly5(cpx *F, int fstride, const cpx *tw, int m)
{
    const cpx ya = tw[fstride * m], yb = tw[fstride * 2 * m];
    cpx *F0 = F, *F1 = F + m, *F2 = F + 2 * m, *F3 = F + 3 * m, *F4 = F + 4 * m;
    for (int u = 0; u < m; u++) {
        cpx s[13];
        s[0] = F0[u];
        C_MUL(s[1], F1[u], tw[u * fstride]); C_MUL(s[2], F2[u], tw[2 * u * fstride]);
        C_MUL(s[3], F3[u], tw[3 * u * fstride]); C_MUL(s[4], F4[u], tw[4 * u * fstride]);
        C_ADD(s[7], s[1], s[4]); C_SUB(s[10], s[1], s[4]);
        C_ADD(s[8], s[2], s[3]); C_SUB(s[9], s[2], s[3]);
        F0[u].r += s[7].r + s[8].r; F0[u].i += s[7].i + s[8].i;
        s[5].r = s[0].r + s[7].r * ya.r + s[8].r * yb.r;
        s[5].i = s[0].i + s[7].i * ya.r + s[8].i * yb.r;
        s[6].r = s[10].i * ya.i + s[9].i * yb.i;
        s[6].i = -(s[10].r * ya.i) - s[9].r * yb.i;
        C_SUB(F1[u], s[5], s[6]); C_ADD(F4[u], s[5], s[6]);
        s[11].r = s[0].r + s[7].r * yb.r + s[8].r * ya.r;
        s[11].i = s[0].i + s[7].i * yb.r + s[8].i * ya.r;
        s[12].r = -(s[10].i * yb.i) + s[9].i * ya.i;
        s[12].i = s[10].r * yb.i - s[9].r * ya.i;
        C_ADD(F2[u], s[11], s[12]); C_SUB(F3[u], s[11], s[12]);
    }
}

static void kf_work(cpx *Fout, const cpx *f, int fstride, const int *fac, const cpx *tw)
{
    cpx *beg = Fout;
    const int p = fac[0], m = fac[1];
    const cpx *end = Fout + p * m;
    if (m == 1) { do { *Fout = *f; f += fstride; } while (++Fout != end); }
    else { do { kf_work(Fout, f, fstride * p, fac + 2, tw); f += fstride; } while ((Fout += m) != end); }
    Fout = beg;
    switch (p) {
    case 2: bfly2(Fout, fstride, tw, m); break;
    case 3: bfly3(Fout, fstride, tw, m); break;
    case 4: bfly4(Fout, fstride, tw, m); break;
    default: bfly5(Fout, fstride, tw, m); break;
    }
}

static int fft_factor(int n, int *fac)
{
    int p = 4, k = 0;
    while (n > 1) {
        while (n % p) {
            if (p == 4) p = 2; else if (p == 2) p = 3; else if (p == 3) p = 5; else return E_CONFIG;
        }
        n /= p;
        if (k >= MAX_FACTORS) return E_CONFIG;
        fac[2 * k] = p; fac[2 * k + 1] = n; k++;
    }
    return k ? E_OK : E_CONFIG;
}

// in-place forward FFT of buf (n complex) using the plan's scratch
static void fft_fwd(itofs_fft_t *P, cpx *buf)
{
    OPC(ITOFS_OPC_FFT, 1);
    kf_work(P->tmp, buf, 1, P->fac, P->tw);
    memcpy(buf, P->tmp, sizeof(cpx) * (size_t)P->n);
}

// ======================================================================================================================
// blob
// ======================================================================================================================
typedef struct {
    char name[64];
    uint32_t dtype, ndim, shape[4], off, nbytes, soff, reserved;
} tent_t;

typedef struct {
    const uint8_t *b; size_t size;
    const char *cfg; uint32_t cfg_len;
    uint32_t n; const uint8_t *table;
} blob_t;

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

static void tent_at(const blob_t *B, uint32_t i, tent_t *t)
{
    const uint8_t *p = B->table + (size_t)i * 104;
    memcpy(t->name, p, 64); t->name[63] = 0;
    t->dtype = rd32(p + 64); t->ndim = rd32(p + 68);
    for (int k = 0; k < 4; k++) t->shape[k] = rd32(p + 72 + 4 * k);
    t->off = rd32(p + 88); t->nbytes = rd32(p + 92); t->soff = rd32(p + 96); t->reserved = rd32(p + 100);
}

static int find(const blob_t *B, const char *name, tent_t *t)
{
    for (uint32_t i = 0; i < B->n; i++) {
        tent_at(B, i, t);
        if (strcmp(t->name, name) == 0) {
            if ((size_t)t->off + t->nbytes > B->size) return E_BLOB;
            if ((t->dtype == 1 || t->dtype == 3) && (size_t)t->soff + 4 * (size_t)t->shape[1] > B->size) return E_BLOB;
            return E_OK;
        }
    }
    return E_MISSING;
}

static int cfg_int(const blob_t *B, const char *key, int *v)
{
    size_t kl = strlen(key);
    const char *p = B->cfg, *end = B->cfg + B->cfg_len;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) nl = end;
        if ((size_t)(nl - p) > kl && memcmp(p, key, kl) == 0 && p[kl] == '=') { *v = atoi(p + kl + 1); return E_OK; }
        p = nl + 1;
    }
    return E_MISSING;
}

static int blob_open(blob_t *B, const void *blob, size_t size)
{
    B->b = (const uint8_t *)blob; B->size = size;
    if (size < 16 || rd32(B->b) != ITF_MAGIC || rd32(B->b + 4) != 1) return E_BLOB;
    B->cfg_len = rd32(B->b + 8);
    B->cfg = (const char *)(B->b + 12);
    if (16 + (size_t)B->cfg_len > size) return E_BLOB;
    B->n = rd32(B->b + 12 + B->cfg_len);
    B->table = B->b + 16 + B->cfg_len;
    if ((size_t)(B->table - B->b) + (size_t)B->n * 104 > size) return E_BLOB;
    return E_OK;
}

int itofs_blob_find(const void *blob, size_t size, const char *name, const void **data, size_t *nbytes, int *dtype)
{
    blob_t B;
    tent_t t;
    int e = blob_open(&B, blob, size);
    if (e) return e;
    if ((e = find(&B, name, &t))) return e;
    *data = B.b + t.off; *nbytes = t.nbytes; *dtype = (int)t.dtype;
    return E_OK;
}

static int g_err;
static char g_err_name[96];
static long g_n8, g_n16, g_n32;

static void set_err(int e, const char *name)
{
    if (!g_err) { g_err = e; snprintf(g_err_name, sizeof g_err_name, "%s", name); }
}

static const float *getf(const blob_t *B, const char *name, long expect)
{
    tent_t t;
    int e = find(B, name, &t);
    if (e) { set_err(e, name); return NULL; }
    if (t.dtype != 0 || (expect >= 0 && (long)(t.nbytes / 4) != expect)) { set_err(E_SHAPE, name); return NULL; }
    g_n32 += t.nbytes / 4;
    return (const float *)(B->b + t.off);
}

static void getq_named(const blob_t *B, itofs_qlin_t *q, const char *wname, const char *bname, int K, int out, int in)
{
    tent_t t;
    int e = find(B, wname, &t);
    memset(q, 0, sizeof *q);
    if (!e && ((t.dtype != 1 && t.dtype != 3) || t.ndim != 3 || (int)t.shape[0] != K || (int)t.shape[1] != out || (int)t.shape[2] != in ||
               t.nbytes != (uint32_t)K * out * in * (t.dtype == 3 ? 2u : 1u))) e = E_SHAPE;
    if (e) { set_err(e, wname); return; }
    q->w = (const int8_t *)(B->b + t.off);
    q->w_lo = t.dtype == 3 ? q->w + (size_t)K * out * in : NULL;   // int16: [hi plane][lo plane]
    q->sw = (const float *)(B->b + t.soff);
    q->K = K; q->out = out; q->in = in;
    q->act16 = (t.dtype == 1 && (t.reserved & 1u)) ? 1 : 0;
    if (t.dtype == 3) g_n16 += t.nbytes / 2; else g_n8 += t.nbytes;
    g_n32 += out;
    if (bname) q->b = getf(B, bname, out);
}

static void getq(const blob_t *B, itofs_qlin_t *q, const char *prefix, int K, int out, int in)
{
    char w[128], b[128];
    snprintf(w, sizeof w, "%s.weight", prefix); snprintf(b, sizeof b, "%s.bias", prefix);
    getq_named(B, q, w, b, K, out, in);
}

static void get_convln(const blob_t *B, itofs_convln_t *L, const char *prefix, int dim, int K, int sdim)   // sdim 0: no FiLM weights
{
    char nm[96];
    snprintf(nm, sizeof nm, "%s.conv", prefix); getq(B, &L->conv, nm, K, dim, dim);
    snprintf(nm, sizeof nm, "%s.norm.weight", prefix); L->ln_g = getf(B, nm, dim);
    snprintf(nm, sizeof nm, "%s.norm.bias", prefix); L->ln_b = getf(B, nm, dim);
    L->film_w = L->film_b = NULL;
    memset(&L->film_q, 0, sizeof L->film_q);
    if (sdim) {
        tent_t t;
        char nb[96];
        snprintf(nm, sizeof nm, "%s.film.weight", prefix); snprintf(nb, sizeof nb, "%s.film.bias", prefix);
        if (!find(B, nm, &t) && (t.dtype == 1 || t.dtype == 3)) { getq_named(B, &L->film_q, nm, nb, 1, 2 * dim, sdim); return; }
        L->film_w = getf(B, nm, (long)2 * dim * sdim);
        L->film_b = getf(B, nb, 2 * dim);
    }
}

int itofs_model_init(itofs_model_t *m, const void *blob, size_t size)
{
    memset(m, 0, sizeof *m);
    blob_t B;
    int be = blob_open(&B, blob, size);
    if (be) return be;

    struct { const char *k; int *v; } keys[] = {
        {"n_vocab", &m->n_vocab}, {"text_dim", &m->text_dim}, {"text_layers", &m->text_layers},
        {"text_kernel", &m->text_kernel}, {"text_rpad", &m->text_rpad}, {"rnn_hidden", &m->rnn_hidden},
        {"style_in", &m->style_in}, {"style_dim", &m->style_dim}, {"dur_layers", &m->dur_layers},
        {"dur_kernel", &m->dur_kernel}, {"dur_rpad", &m->dur_rpad}, {"sent_pos", &m->sent_pos}, {"hf_dim", &m->hf_dim},
        {"pros_dim", &m->pros_dim}, {"pros_layers", &m->pros_layers}, {"pros_kernel", &m->pros_kernel},
        {"pros_rpad", &m->pros_rpad}, {"n_pros_out", &m->n_pros_out}, {"dec_in", &m->dec_in}, {"dec_dim", &m->dec_dim},
        {"embed_kernel", &m->embed_kernel}, {"dec_inter", &m->dec_inter}, {"dec_blocks", &m->dec_blocks},
        {"dec_kernel", &m->dec_kernel}, {"dec_rpad", &m->dec_rpad}, {"harm_in", &m->harm_in}, {"dec_out", &m->dec_out_dim},
        {"n_fft", &m->n_fft}, {"hop", &m->hop}, {"sr", &m->sr}, {"n_harm", &m->n_harm}, {"fps", &m->fps},
        {"n_styles", &m->n_styles},
    };
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++)
        if (cfg_int(&B, keys[i].k, keys[i].v)) { fprintf(stderr, "itofs_model_init: missing key %s\n", keys[i].k); return E_MISSING; }
    if (cfg_int(&B, "rnn_bidir", &m->rnn_bidir)) m->rnn_bidir = 1;   // blobs before the key: bidirectional
    if (cfg_int(&B, "arch", &m->arch)) m->arch = 2;                    // blobs before the key: v2
    if (m->arch != 2 && m->arch != 3) return E_CONFIG;
    if (m->arch == 3) {
        struct { const char *k; int *v; } k3[] = { {"n_mels", &m->n_mels}, {"mel_dim", &m->mel_dim}, {"mel_layers", &m->mel_layers},
                                                    {"mel_kernel", &m->mel_kernel}, {"mel_rpad", &m->mel_rpad} };
        for (size_t i = 0; i < sizeof k3 / sizeof k3[0]; i++)
            if (cfg_int(&B, k3[i].k, k3[i].v)) { fprintf(stderr, "itofs_model_init: missing key %s\n", k3[i].k); return E_MISSING; }
        if (m->mel_layers < 1 || m->mel_layers > ITOFS_MAX_LAYERS || m->mel_rpad < 0 || m->mel_rpad > m->mel_kernel - 1 ||
            m->n_mels < 1 || m->mel_dim < 1)
            return E_CONFIG;
    }
    int fac[2 * MAX_FACTORS];
    if (m->text_layers > ITOFS_MAX_LAYERS || m->dur_layers > ITOFS_MAX_LAYERS || m->pros_layers > ITOFS_MAX_LAYERS ||
        m->text_layers < 1 || m->dur_layers < 1 || m->pros_layers < 1 ||
        m->dec_blocks > ITOFS_MAX_BLOCKS || m->dec_blocks < 1 || m->hf_dim != m->text_dim + 2 + (m->sent_pos ? 1 : 0) ||
        m->dec_in != (m->arch == 3 ? m->n_mels + 2 : m->hf_dim + 3) || m->harm_in != m->n_fft + 2 || m->dec_out_dim != m->n_fft + 2 || m->n_pros_out != 3 ||
        (m->hop & 1) || (m->n_fft & 1) || m->n_fft < 2 * m->hop || fft_factor(m->n_fft / 2, fac) ||
        m->text_rpad < 0 || m->text_rpad > m->text_kernel - 1 || m->dur_rpad < 0 || m->dur_rpad > m->dur_kernel - 1 ||
        m->pros_rpad < 0 || m->pros_rpad > m->pros_kernel - 1 || m->dec_rpad < 0 || m->dec_rpad > m->dec_kernel - 1 ||
        m->dec_rpad > m->embed_kernel - 1 || m->n_harm < 1 || m->n_styles < 1)
        return E_CONFIG;

    g_err = 0; g_n8 = 0; g_n16 = 0; g_n32 = 0; g_err_name[0] = 0;
    char nm[96], nm2[96];
    const int D = m->text_dim, S = m->style_dim, PD = m->pros_dim, DD = m->dec_dim, H = m->rnn_hidden;
    m->stats = getf(&B, "stats", 4);
    const float *cst = getf(&B, "consts", 8);
    if (cst) { m->src_amp = cst[0]; m->src_noise = cst[1]; m->uv_hz = cst[2]; m->mag_max = cst[3]; m->eps_text = cst[4]; m->eps_dec = cst[5];
               m->voc_hz = cst[6]; }
    m->emb = getf(&B, "emb.weight", (long)m->n_vocab * D);
    {
        tent_t tt;
        if (!find(&B, "style_table", &tt)) {
            m->style_table_w = 2 * D * m->dur_layers + 2 * PD * m->pros_layers + (m->arch == 3 ? 2 * m->mel_dim * m->mel_layers : 0);
            if (m->arch != 3) { set_err(E_CONFIG, "style_table needs arch 3"); }
            m->style_table = getf(&B, "style_table", (long)m->n_styles * m->style_table_w);
        }
    }
    const int SF = m->style_table ? 0 : S;           // FiLM weights are only loaded without a style table
    {
        tent_t tt;
        m->has_sp = m->arch == 3 && !find(&B, "sp.net.0.weight", &tt);
        if (m->has_sp) {
            getq_named(&B, &m->sp[0], "sp.net.0.weight", "sp.net.0.bias", 1, (int)tt.shape[1], 2 * D + 1);
            const int hid = m->sp[0].out;
            getq_named(&B, &m->sp[1], "sp.net.2.weight", "sp.net.2.bias", 1, hid, hid);
            getq_named(&B, &m->sp[2], "sp.net.4.weight", "sp.net.4.bias", 1, m->style_in, hid);
            m->sp_mu = getf(&B, "sp.mu", m->style_in);
            if (m->style_table) set_err(E_CONFIG, "style predictor and style table");
        }
    }
    if (!m->style_table) {
        tent_t tt;
        if (!m->has_sp || !find(&B, "style_bank", &tt)) m->style_bank = getf(&B, "style_bank", (long)m->n_styles * m->style_in);
        if (!find(&B, "style_proj.weight", &tt) && (tt.dtype == 1 || tt.dtype == 3))
            getq_named(&B, &m->style_q, "style_proj.weight", "style_proj.bias", 1, S, m->style_in);
        else {
            m->style_w = getf(&B, "style_proj.weight", (long)S * m->style_in);
            m->style_b = getf(&B, "style_proj.bias", S);
        }
    }
    for (int l = 0; l < m->text_layers; l++) { snprintf(nm, sizeof nm, "enc.%d", l); get_convln(&B, &m->enc[l], nm, D, m->text_kernel, 0); }
    if (H) {
        for (int d = 0; d < (m->rnn_bidir ? 2 : 1); d++) {
            const char *sfx = d ? "_reverse" : "";
            snprintf(nm, sizeof nm, "rnn.weight_ih_l0%s", sfx); snprintf(nm2, sizeof nm2, "rnn.bias_ih_l0%s", sfx);
            getq_named(&B, &m->rnn_ih[d], nm, nm2, 1, 3 * H, D);
            snprintf(nm, sizeof nm, "rnn.weight_hh_l0%s", sfx); m->rnn_hh[d] = getf(&B, nm, (long)3 * H * H);
            snprintf(nm, sizeof nm, "rnn.bias_hh_l0%s", sfx); m->rnn_bhh[d] = getf(&B, nm, 3 * H);
        }
        getq(&B, &m->rnn_proj, "rnn_proj", 1, D, (m->rnn_bidir ? 2 : 1) * H);
    }
    for (int l = 0; l < m->dur_layers; l++) { snprintf(nm, sizeof nm, "dur_layers.%d", l); get_convln(&B, &m->dur[l], nm, D, m->dur_kernel, SF); }
    m->dur_out_w = getf(&B, "dur_out.weight", D);
    m->dur_out_b = getf(&B, "dur_out.bias", 1);
    getq(&B, &m->pros_in, "pros_in", 1, PD, m->hf_dim);
    for (int l = 0; l < m->pros_layers; l++) { snprintf(nm, sizeof nm, "pros.%d", l); get_convln(&B, &m->pros[l], nm, PD, m->pros_kernel, SF); }
    m->pros_out_w = getf(&B, "pros_out.weight", (long)3 * PD);
    m->pros_out_b = getf(&B, "pros_out.bias", 3);
    if (m->arch == 3) {
        m->f0stats = getf(&B, "dec.f0stats", 2);
        m->mel_mean = getf(&B, "mel_mean", m->n_mels);
        m->mel_std = getf(&B, "mel_std", m->n_mels);
        getq(&B, &m->mel_in, "mel_in", 1, m->mel_dim, m->hf_dim + 3);
        for (int l = 0; l < m->mel_layers; l++) { snprintf(nm, sizeof nm, "mel_blocks.%d", l); get_convln(&B, &m->mel[l], nm, m->mel_dim, m->mel_kernel, SF); }
        getq(&B, &m->mel_out, "mel_out", 1, m->n_mels, m->mel_dim);
    }
    getq(&B, &m->dec_embed, "dec.embed", m->embed_kernel, DD, m->dec_in);
    if (m->arch == 2) {                              // v3's vocoder has no style conditioning
        m->cond_w = getf(&B, "dec.cond.weight", (long)DD * S);
        m->cond_b = getf(&B, "dec.cond.bias", DD);
    }
    m->n0_g = getf(&B, "dec.norm0.weight", DD); m->n0_b = getf(&B, "dec.norm0.bias", DD);
    getq(&B, &m->harm_proj, "dec.harm_proj", 1, DD, m->harm_in);
    for (int b = 0; b < m->dec_blocks; b++) {
        itofs_block_t *bk = &m->blk[b];
        snprintf(nm, sizeof nm, "dec.blocks.%d.dw.weight", b); bk->dw_w = getf(&B, nm, (long)DD * m->dec_kernel);
        snprintf(nm, sizeof nm, "dec.blocks.%d.dw.bias", b); bk->dw_b = getf(&B, nm, DD);
        snprintf(nm, sizeof nm, "dec.blocks.%d.norm.weight", b); bk->ln_g = getf(&B, nm, DD);
        snprintf(nm, sizeof nm, "dec.blocks.%d.norm.bias", b); bk->ln_b = getf(&B, nm, DD);
        snprintf(nm, sizeof nm, "dec.blocks.%d.gamma", b); bk->gamma = getf(&B, nm, DD);
        snprintf(nm, sizeof nm, "dec.blocks.%d.pw1", b); getq(&B, &bk->pw1, nm, 1, m->dec_inter, DD);
        snprintf(nm, sizeof nm, "dec.blocks.%d.pw2", b); getq(&B, &bk->pw2, nm, 1, DD, m->dec_inter);
    }
    m->n1_g = getf(&B, "dec.norm1.weight", DD); m->n1_b = getf(&B, "dec.norm1.bias", DD);
    getq(&B, &m->dec_out, "dec.out", 1, m->dec_out_dim, DD);
    m->blob_size = size;
    m->n_int8_params = g_n8; m->n_int16_params = g_n16; m->n_f32_params = g_n32;
    if (g_err) { fprintf(stderr, "itofs_model_init: %s: %s\n", itofs_strerror(g_err), g_err_name); return g_err; }
    return E_OK;
}

// ======================================================================================================================
// default GEMM
// ======================================================================================================================
void itofs_qgemm_ref(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user)
{
    (void)user;
    int o = 0;
    for (; o + 4 <= out; o += 4) {
        const int8_t *w0 = w + (size_t)o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
        int r = 0;
        for (; r + 2 <= rows; r += 2) {
            const int8_t *x0 = x + (size_t)r * ldx, *x1 = x0 + ldx;
            int32_t a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
            for (int i = 0; i < in; i++) {
                int32_t p = x0[i], q = x1[i], c0 = w0[i], c1 = w1[i], c2 = w2[i], c3 = w3[i];
                a00 += p * c0; a01 += p * c1; a02 += p * c2; a03 += p * c3;
                a10 += q * c0; a11 += q * c1; a12 += q * c2; a13 += q * c3;
            }
            int32_t *y0 = acc + (size_t)r * out + o, *y1 = y0 + out;
            y0[0] = a00; y0[1] = a01; y0[2] = a02; y0[3] = a03;
            y1[0] = a10; y1[1] = a11; y1[2] = a12; y1[3] = a13;
        }
        for (; r < rows; r++) {
            const int8_t *x0 = x + (size_t)r * ldx;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int i = 0; i < in; i++) { int32_t p = x0[i]; a0 += p * w0[i]; a1 += p * w1[i]; a2 += p * w2[i]; a3 += p * w3[i]; }
            int32_t *y0 = acc + (size_t)r * out + o;
            y0[0] = a0; y0[1] = a1; y0[2] = a2; y0[3] = a3;
        }
    }
    for (; o < out; o++) {
        const int8_t *w0 = w + (size_t)o * in;
        for (int r = 0; r < rows; r++) {
            const int8_t *x0 = x + (size_t)r * ldx;
            int32_t a = 0;
            for (int i = 0; i < in; i++) a += (int32_t)x0[i] * w0[i];
            acc[(size_t)r * out + o] = a;
        }
    }
}

// ======================================================================================================================
// stages
// ======================================================================================================================
enum { SK_PIN, SK_PROS, SK_CUR, SK_SRC, SK_HFT, SK_EMB, SK_BLK, SK_HEAD, SK_OLA, SK_MIN, SK_MEL, SK_MOUT };
#define MAX_STAGES (2 * ITOFS_MAX_LAYERS + ITOFS_MAX_BLOCKS + 10)

struct itofs_stage {
    int kind, idx;
    int hfR;                       // reach into the length-regulated text features (-1: does not read them)
    int np, prod[2], L[2], R[2];   // producer stages and reach (the length-regulated text features are implicit)
    int width;                     // floats per frame in this stage's ring
    int rframes;                   // ring length in frames (power of two; see stage_rings)
    int bmax;                      // most rows one run may compute (hot scratch capacity; >= chunk_frames)
    float *ring;
    int done;                      // frames [0, done) computed
};

enum { TK_ENC, TK_GRU, TK_PROJ, TK_DUR, TK_DOUT };
#define MAX_TSTAGES (2 * ITOFS_MAX_LAYERS + 4)
struct itofs_tstage { int kind, idx, L, R, width, done, bmax; float *ring; };

static int build_tgraph(const itofs_model_t *m, itofs_tstage_t *t)
{
    int n = 0;
#define TADD(k_, i_, l_, r_, w_) do { if (t) { t[n].kind = k_; t[n].idx = i_; t[n].L = l_; t[n].R = r_; t[n].width = w_; t[n].done = 0; t[n].ring = NULL; } n++; } while (0)
    for (int l = 0; l < m->text_layers; l++) TADD(TK_ENC, l, m->text_kernel - 1 - m->text_rpad, m->text_rpad, m->text_dim);
    if (m->rnn_hidden) TADD(TK_GRU, 0, 0, 0, m->rnn_hidden);
    TADD(TK_PROJ, 0, 0, 0, 0);                         // writes c->h (all tokens: the frame side reads it)
    for (int l = 0; l < m->dur_layers; l++) TADD(TK_DUR, l, m->dur_kernel - 1 - m->dur_rpad, m->dur_rpad, m->text_dim);
    TADD(TK_DOUT, 0, 0, 0, 0);
#undef TADD
    return n;
}

static int text_lookahead(const itofs_model_t *m)
{
    return m->text_layers * m->text_rpad + m->dur_layers * m->dur_rpad;
}

typedef struct { uint8_t *base; size_t used; } bump_t;
static void *bump(bump_t *a, size_t bytes)
{
    size_t off = (a->used + 15) & ~(size_t)15;
    a->used = off + bytes;
    return a->base ? (void *)(a->base + off) : NULL;
}

// stage graph (shared by the sizing pass and init)
static int build_graph(const itofs_model_t *m, itofs_stage_t *st, int *hft_L, int *hft_R, int *ola_L, int *ola_R)
{
    const int hop = m->hop, N = m->n_fft;
    // harmonic STFT frame t reads source samples [t*hop - N/2 - hop/2, t*hop + N/2 - hop/2)
    *hft_L = (N / 2 + hop / 2 + hop - 1) / hop;
    *hft_R = floordiv(N / 2 - hop / 2 - 1, hop);
    // iSTFT: output sample p = t*hop + s + N/2 + hop/2 gets frames f with f*hop <= p < f*hop + N
    int L = 0, R = 0, t = 1000, off = N / 2 + hop / 2;
    for (int s = 0; s < hop; s++) {
        int p = t * hop + s + off;
        L = imax(L, t - floordiv(p - N + hop, hop)); R = imax(R, floordiv(p, hop) - t);
    }
    *ola_L = L; *ola_R = R;
    int n = 0;
#define ADD(kind_, idx_, w_) do { if (st) { memset(&st[n], 0, sizeof st[n]); st[n].kind = kind_; st[n].idx = idx_; st[n].width = w_; st[n].hfR = -1; } n++; } while (0)
#define HF(r_) do { if (st) st[n - 1].hfR = r_; } while (0)
#define EDGE(p_, l_, r_) do { if (st) { itofs_stage_t *q_ = &st[n - 1]; q_->prod[q_->np] = p_; q_->L[q_->np] = l_; q_->R[q_->np] = r_; q_->np++; } } while (0)
    ADD(SK_PIN, 0, m->pros_dim); HF(0);
    const int pl = m->pros_kernel - 1 - m->pros_rpad, pr = m->pros_rpad;
    for (int l = 0; l < m->pros_layers; l++) { ADD(SK_PROS, l, m->pros_dim); EDGE(n - 2, pl, pr); }
    // CUR row: [lf0n, voiced, energy, source F0 (Hz)] (+ arch 3: [vocoder lf0n, vocoder voiced])
    const int s_cur = n; ADD(SK_CUR, 0, m->arch == 3 ? 6 : 4); EDGE(n - 2, 0, 0);
    int s_dec_in = s_cur;                    // producer of the decoder embed's input frames
    if (m->arch == 3) {                      // mel head: MIN (1x1 on [text features | curves]) -> MEL x L -> MOUT
        ADD(SK_MIN, 0, m->mel_dim); EDGE(s_cur, 0, 0); HF(0);
        const int ml = m->mel_kernel - 1 - m->mel_rpad, mr = m->mel_rpad;
        for (int l = 0; l < m->mel_layers; l++) { ADD(SK_MEL, l, m->mel_dim); EDGE(n - 2, ml, mr); }
        s_dec_in = n; ADD(SK_MOUT, 0, m->n_mels + 2); EDGE(n - 2, 0, 0); EDGE(s_cur, 0, 0);
    }
    const int s_src = n; ADD(SK_SRC, 0, hop); EDGE(s_cur, 1, 1);
    const int s_hft = n; ADD(SK_HFT, 0, m->dec_dim); EDGE(s_src, *hft_L, *hft_R);
    ADD(SK_EMB, 0, m->dec_dim); EDGE(s_dec_in, m->embed_kernel - 1 - m->dec_rpad, m->dec_rpad); EDGE(s_hft, 0, 0);
    if (m->arch == 2) HF(m->dec_rpad);
    const int bl = m->dec_kernel - 1 - m->dec_rpad, br = m->dec_rpad;
    for (int b = 0; b < m->dec_blocks; b++) { ADD(SK_BLK, b, m->dec_dim); EDGE(n - 2, bl, br); }
    ADD(SK_HEAD, 0, N); EDGE(n - 2, 0, 0);
    ADD(SK_OLA, 0, 0); EDGE(n - 2, L, R);
    (void)s_src;
#undef ADD
#undef EDGE
#undef HF
    return n;
}

// Ring length of every stage. D[s] = largest total right reach from stage s to the output, so an advance() to output
// frame F needs stage s up to F + D[s]. A producer p is at most nb + D[p] frames ahead of the start of the call, and a
// consumer c (edge reach L) still reads rows from (its previous position - L) = (F - nb + D[c] - L): the ring must hold
// max(nb + D[p], max_c (nb + D[p] - D[c] + L)) rows (+2 margin). The streaming test (chunks 1..32 vs one whole chunk,
// bit-identical) checks this sizing.
static void stage_rings(const itofs_model_t *m, int C, int *rf, int *la_out)
{
    itofs_stage_t st[MAX_STAGES];
    int a, b, c, d;
    const int S = build_graph(m, st, &a, &b, &c, &d);
    int D[MAX_STAGES] = {0};
    for (int s = S - 1; s >= 0; s--)
        for (int e = 0; e < st[s].np; e++) {
            int p = st[s].prod[e];
            D[p] = imax(D[p], D[s] + st[s].R[e]);
        }
    for (int p = 0; p < S; p++) {
        int need = C + D[p] + 2;
        for (int s = 0; s < S; s++)
            for (int e = 0; e < st[s].np; e++)
                if (st[s].prod[e] == p) need = imax(need, C + D[p] - D[s] + st[s].L[e] + 2);
        int r = 1;
        while (r < need) r <<= 1;
        if (rf) rf[p] = r;
    }
    int la = 0;
    for (int s = 0; s < S; s++) if (st[s].hfR >= 0) la = imax(la, D[s] + st[s].hfR);
    if (la_out) *la_out = la;
}

static int ring_frames(const itofs_model_t *m, int C, int *maxD_out)
{
    itofs_stage_t st[MAX_STAGES];
    int a, b, c, d;
    const int S = build_graph(m, st, &a, &b, &c, &d);
    int D[MAX_STAGES] = {0}, maxD = 0, maxL = 0;
    for (int s = S - 1; s >= 0; s--)
        for (int e = 0; e < st[s].np; e++) {
            int p = st[s].prod[e];
            D[p] = imax(D[p], D[s] + st[s].R[e]);
            maxL = imax(maxL, st[s].L[e]);
        }
    for (int s = 0; s < S; s++) maxD = imax(maxD, D[s]);
    // the text features feed PIN (reach 0) and EMB (v2, embed reach) or MIN (v3, reach 0); total lookahead in frames
    int la = 0;
    for (int s = 0; s < S; s++) if (st[s].hfR >= 0) la = imax(la, D[s] + st[s].hfR);
    if (maxD_out) *maxD_out = la;
    int need = C + maxD + maxL + 2, r = 1;
    while (r < need) r <<= 1;
    return r;
}

static int max_qout_tile, max_k;
static long max_qrow;          // bytes of quantised activations per row, narrow layers (all but harm_proj)
static int max_qin_narrow;     // widest narrow input (a 16-bit-activation run on an arena sized for 8 bits needs two planes of it)
static int g_pa;               // activation planes of the int8-weight layers the arena is sized for
static void scan_q(const itofs_qlin_t *q, int wide)
{
    if (!q->w) return;
    if (!wide) { const int P = (q->w_lo || q->act16) ? 2 : g_pa; max_qrow = (long)imax((int)max_qrow, P * pad16(q->in)); max_qin_narrow = imax(max_qin_narrow, q->in); }
    max_qout_tile = imax(max_qout_tile, imin(q->out, ACC_TILE));
    max_k = imax(max_k, q->K);
}

static void layout(itofs_ctx_t *c, const itofs_model_t *m, const itofs_limits_t *lim, bump_t *hot, bump_t *bulk)
{
    const int D = m->text_dim, H = m->rnn_hidden, PD = m->pros_dim, DD = m->dec_dim, N = m->n_fft, M = N / 2;
    const int Lmax = lim->max_tokens, nb = lim->chunk_frames;
    max_qout_tile = max_k = 0; max_qrow = 0; max_qin_narrow = 0; g_pa = lim->act_bits == 16 ? 2 : 1;
    for (int l = 0; l < m->text_layers; l++) scan_q(&m->enc[l].conv, 0);
    for (int l = 0; l < m->dur_layers; l++) scan_q(&m->dur[l].conv, 0);
    for (int l = 0; l < m->pros_layers; l++) scan_q(&m->pros[l].conv, 0);
    if (H) { scan_q(&m->rnn_ih[0], 0); scan_q(&m->rnn_ih[1], 0); scan_q(&m->rnn_proj, 0); }
    scan_q(&m->pros_in, 0); scan_q(&m->dec_embed, 0); scan_q(&m->harm_proj, 1); scan_q(&m->dec_out, 0);
    if (m->arch == 3) { scan_q(&m->mel_in, 0); scan_q(&m->mel_out, 0); for (int l = 0; l < m->mel_layers; l++) { scan_q(&m->mel[l].conv, 0); scan_q(&m->mel[l].film_q, 0); } }
    for (int i = 0; i < 3 && m->has_sp; i++) scan_q(&m->sp[i], 0);
    scan_q(&m->style_q, 0);
    for (int l = 0; l < m->dur_layers; l++) scan_q(&m->dur[l].film_q, 0);
    for (int l = 0; l < m->pros_layers; l++) scan_q(&m->pros[l].film_q, 0);
    const int MD = m->arch == 3 ? m->mel_dim : 0, MIN_IN = m->arch == 3 ? m->hf_dim + 3 : 0, NM = m->arch == 3 ? m->n_mels : 0;
    for (int b = 0; b < m->dec_blocks; b++) { scan_q(&m->blk[b].pw1, 0); scan_q(&m->blk[b].pw2, 0); }
    max_k = imax(max_k, m->dec_kernel);             // the depthwise convs gather K rows too
    const int grow = nb + max_k - 1;
    const int whole_txt = (m->rnn_hidden && m->rnn_bidir) || m->sent_pos || lim->whole_text;
    const int tb = whole_txt ? imax(nb, TEXT_BATCH) : nb;                       // whole-sentence text-side batch
    const int tk = imax(imax(m->text_kernel, m->dur_kernel), 1), growt = tb + tk - 1, tin = imax(D, 2 * H);
    const int gw = imax(imax(imax(imax(PD, m->dec_in), imax(DD, m->hf_dim)), imax(D, H)), imax(MD, MIN_IN));
    // y0: prosody / embed / block rows (dw-conv output AND pw2 output: the block's dw output is dead once pw1 has quantised it), the
    //     GRU gates, the mel rows, and WIDE_ROWS rows of the wide (n_fft + 2) STFT frames / head output; its harm_proj output (256
    //     wide) also lands here: the wide input is dead once quantised.
    // y1: one STFT frame, the ConvNeXt hidden (nb x inter) and one half spectrum.
    const int nbf = nb + FILL_ROWS;
    const int y0w = imax(imax(imax(imax(DD, PD), imax(D, 3 * H)), imax(MD, NM)), 1), y1n = imax((nb + 1) * m->dec_inter, N + 2);
    // ---- hot scratch
    c->s_g0 = bump(hot, (size_t)(grow + 1) * gw * 4);
    // s_g1 (style predictor, once per utterance before any frame) is the second core's FFT scratch: M complex + N + 8 + N + 2 floats
    const size_t g1n = imax(3 * N + 16, m->has_sp ? m->sp[0].out : 0);
    c->s_g1 = bump(hot, g1n * 4);
    const long y0n = imax((long)nbf * y0w, (long)WIDE_ROWS * (N + 2));
    c->s_y0 = bump(hot, (size_t)y0n * 4);
    c->s_y1 = bump(hot, (size_t)y1n * 4);
    const int qs_rows = imax(imax(grow, growt), MAX_FILL + max_k);
    c->s_qs = bump(hot, (size_t)qs_rows * 4);
    // (c->act_bits may be raised to 16 later: even then one row of the widest block layer, with its halo, must fit)
    const long q8_rows = imax((int)((long)grow * max_qrow), (int)((long)(1 + max_k) * PLANES * pad16(max_qin_narrow))), q8_text = (long)growt * PLANES * pad16(tin), q8_wide = (long)WIDE_ROWS * g_pa * pad16(N + 2);
    c->cap_q8 = imax((int)imax((int)q8_rows, (int)q8_text), (int)q8_wide);
    c->s_q8 = bump(hot, (size_t)c->cap_q8 + 64);
    c->acc_bytes = ACC_BYTES;
    c->cap_g0 = (long)(grow + 1) * gw; c->cap_y0 = y0n; c->cap_y1 = y1n; c->cap_qs = qs_rows;
    c->s_acc = bump(hot, c->acc_bytes);
    c->fbuf = bump(hot, (size_t)(N + 8) * 4);
    c->win = bump(hot, (size_t)N * 4);
    c->fft = bump(hot, sizeof(itofs_fft_t));
    cpx *tmp = bump(hot, sizeof(cpx) * (size_t)M);
    // second core's FFT scratch (row-parallel HFT / HEAD), inside s_g1
    c->fft_b = bump(hot, sizeof(itofs_fft_t));
    cpx *tmp_b = c->s_g1 ? (cpx *)c->s_g1 : NULL;                      // M complex
    c->fbuf_b = c->s_g1 ? c->s_g1 + 2 * M : NULL;                         // N + 8
    c->x_b = c->s_g1 ? c->s_g1 + 2 * M + N + 8 : NULL;                    // N + 2
    c->tb = tb;
    // read-only tables (read sequentially, cache friendly): bulk, to keep the hot arena in internal SRAM
    c->win2 = bump(bulk, (size_t)N * 4);
    c->rtw = bump(bulk, (size_t)(M + 1) * 2 * 4);
    cpx *tw = bump(bulk, sizeof(cpx) * (size_t)M);
    if (c->fft) { c->fft->tw = tw; c->fft->tmp = tmp; c->fft_b->tw = tw; c->fft_b->tmp = tmp_b; }
    c->src_anc = bump(bulk, (size_t)2 * imax(nb, MAX_FILL) * sizeof(double));
    c->src_w = bump(bulk, (size_t)m->hop * sizeof(float));
    // ---- bulk: text side and the int16 conversion buffer
    c->s_pcm = bump(bulk, (size_t)nb * m->hop * 4);
    c->dur = bump(bulk, (size_t)Lmax * 4);
    c->start = bump(bulk, (size_t)(Lmax + 1) * 4);
    c->h = bump(bulk, (size_t)Lmax * D * 4);
    c->logd = bump(bulk, (size_t)Lmax * 4);
    // whole-sentence text side buffers: needed by bidirectional / sent_pos models, or on request (lim->whole_text)
    const int whole = (m->rnn_hidden && m->rnn_bidir) || m->sent_pos || lim->whole_text;
    c->t_x = whole ? bump(bulk, (size_t)Lmax * D * 4) : NULL;
    // t_y (conv / rnn_proj outputs) and t_g (GRU input gates, live only inside the recurrence) never overlap in time
    c->t_y = whole ? bump(bulk, (size_t)Lmax * imax(D, 3 * H) * 4) : NULL;
    c->t_g = c->t_y;
    c->t_o = whole ? bump(bulk, (size_t)Lmax * imax(2 * H, 1) * 4) : NULL;
    c->t_pad = bump(bulk, (size_t)imax(grow, growt) * tin * 4);
    c->t_film = bump(bulk, (size_t)imax(imax(imax(2 * D, 2 * PD), 5 * H + 4), 2 * MD) * 4);
    c->s = bump(bulk, (size_t)m->style_dim * 4);
    c->cond = bump(bulk, (size_t)DD * 4);
    for (int l = 0; l < m->pros_layers; l++) { c->film_g1[l] = bump(bulk, (size_t)PD * 4); c->film_b[l] = bump(bulk, (size_t)PD * 4); }
    for (int l = 0; l < m->dur_layers; l++) { c->dfilm_g1[l] = bump(bulk, (size_t)D * 4); c->dfilm_b[l] = bump(bulk, (size_t)D * 4); }
    for (int l = 0; l < (m->arch == 3 ? m->mel_layers : 0); l++) { c->mfilm_g1[l] = bump(bulk, (size_t)MD * 4); c->mfilm_b[l] = bump(bulk, (size_t)MD * 4); }
    c->tok = bump(bulk, (size_t)Lmax * 4);
    c->gru_h = bump(bulk, (size_t)imax(H, 1) * 4);
    // ---- token stages (incremental text side) and their rings
    {
        itofs_tstage_t *ts = bump(bulk, sizeof(itofs_tstage_t) * MAX_TSTAGES);
        c->tst = ts;
        itofs_tstage_t tmp_t[MAX_TSTAGES];
        const int NT = build_tgraph(m, ts ? ts : tmp_t);
        c->n_tst = NT;
        c->text_la = text_lookahead(m);
        int maxL = 0, r = 1;
        for (int i = 0; i < NT; i++) maxL = imax(maxL, (ts ? ts : tmp_t)[i].L);
        while (r < nb + c->text_la + maxL + 2) r <<= 1;
        c->tring = r; c->tring_mask = r - 1;
        for (int i = 0; i < NT; i++) {
            const int w = (ts ? ts : tmp_t)[i].width;
            float *rg = w ? bump(bulk, (size_t)r * w * 4) : NULL;
            if (ts) ts[i].ring = rg;
        }
    }
    // ---- stages and rings
    int hl, hr, ol, orr;
    itofs_stage_t *st = bump(bulk, sizeof(itofs_stage_t) * MAX_STAGES);
    c->st = st;
    const int S = build_graph(m, st, &hl, &hr, &ol, &orr);
    c->n_stages = S;
    const int ring = ring_frames(m, nb, NULL);
    int rf[MAX_STAGES];
    stage_rings(m, nb, rf, NULL);
    for (int i = 0; i < S; i++) {
        int w = 0;
        if (st) w = st[i].width;
        else {   // dry run: rebuild widths
            itofs_stage_t tmpst[MAX_STAGES];
            build_graph(m, tmpst, &hl, &hr, &ol, &orr);
            w = tmpst[i].width;
        }
        float *r = w ? bump(bulk, (size_t)rf[i] * w * 4) : NULL;
        if (st) { st[i].ring = r; st[i].rframes = rf[i]; }
    }
    c->ring = ring; c->ring_mask = ring - 1;
    c->ola_L = ol; c->ola_R = orr; c->ola_off = N / 2 + m->hop / 2;
    c->nb = nb;
    c->acc_tile = ACC_TILE;
}

void itofs_arena_bytes(const itofs_model_t *m, const itofs_limits_t *lim, size_t *hot_bytes, size_t *bulk_bytes)
{
    itofs_ctx_t tmp;
    memset(&tmp, 0, sizeof tmp);
    bump_t h = {NULL, 0}, b = {NULL, 0};
    layout(&tmp, m, lim, &h, &b);
    *hot_bytes = h.used + 16;
    *bulk_bytes = b.used + 16;
}

int itofs_init(itofs_ctx_t *c, const itofs_model_t *m, const itofs_limits_t *lim,
               void *hot, size_t hot_bytes, void *bulk, size_t bulk_bytes)
{
    if (!c || !m || !lim || lim->max_tokens < 1 || lim->chunk_frames < 1 || (lim->act_bits && lim->act_bits != 8 && lim->act_bits != 16))
        return E_ARG;
    size_t hb, bb;
    itofs_arena_bytes(m, lim, &hb, &bb);
    if (hot_bytes < hb || bulk_bytes < bb) return E_ARENA;
    memset(c, 0, sizeof *c);
    c->m = m; c->lim = *lim;
    c->qgemm = itofs_qgemm_ref;
    c->act_bits = lim->act_bits ? lim->act_bits : 8;
    c->act_planes = c->act_bits == 16 ? 2 : 1;
    uintptr_t ha = ((uintptr_t)hot + 15) & ~(uintptr_t)15, ba = ((uintptr_t)bulk + 15) & ~(uintptr_t)15;
    bump_t h = {(uint8_t *)ha, 0}, b = {(uint8_t *)ba, 0};
    layout(c, m, lim, &h, &b);
    c->hot_used = h.used; c->bulk_used = b.used;
    // tables (double-precision sin/cos from + - * / only: identical on every IEEE target)
    const int N = m->n_fft, M = N / 2;
    for (int n = 0; n < N; n++) {
        double s, co; itf_sincos2pi_d(n, N, &s, &co);
        c->win[n] = (float)(0.5 - 0.5 * co);          // periodic Hann
        c->win2[n] = c->win[n] * c->win[n];
    }
    for (int k = 0; k <= M; k++) {
        double s, co; itf_sincos2pi_d(k, N, &s, &co);
        c->rtw[2 * k] = (float)co; c->rtw[2 * k + 1] = (float)s;
    }
    for (int j = 0; j < m->hop; j++) c->src_w[j] = ((float)j + 0.5f) / (float)m->hop;
    c->fft->n = M;
    if (fft_factor(M, c->fft->fac)) return E_CONFIG;
    c->fft_b->n = M;
    memcpy(c->fft_b->fac, c->fft->fac, sizeof c->fft->fac);
    for (int k = 0; k < M; k++) {
        double s, co; itf_sincos2pi_d(k, M, &s, &co);
        c->fft->tw[k].r = (float)co; c->fft->tw[k].i = (float)-s;   // exp(-2 pi i k / M)
    }
    return E_OK;
}

// ======================================================================================================================
// primitives
// ======================================================================================================================
// quantise one row of `in` floats: P = 2: 15 bits split exactly into two int8 planes (hi at q, lo at q + ldq,
// q = 128*hi + lo); P = 1: one int8 plane, q = floor(x * 127 / max|x| + 0.5). Returns the scale. The bytes between `in` and the
// 16-byte row stride ldq are set to zero (the S3 GEMM kernel runs whole vectors over them).

static float quant_row(const float *x, int in, int ldq, int P, int8_t *q)
{
    OPC(ITOFS_OPC_QUANT, in);
    float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;                // max |x| in four independent chains (order-free: the max is exact)
    int i = 0;
    for (; i + 4 <= in; i += 4) {
        const float a0 = fabsf(x[i]), a1 = fabsf(x[i + 1]), a2 = fabsf(x[i + 2]), a3 = fabsf(x[i + 3]);
        if (a0 > m0) m0 = a0;
        if (a1 > m1) m1 = a1;
        if (a2 > m2) m2 = a2;
        if (a3 > m3) m3 = a3;
    }
    for (; i < in; i++) { const float a = fabsf(x[i]); if (a > m0) m0 = a; }
    if (m1 > m0) m0 = m1;
    if (m3 > m2) m2 = m3;
    if (m2 > m0) m0 = m2;
    const float mx = m0;
    if (!(mx > 0.f)) { memset(q, 0, (size_t)ldq * P); return 0.f; }
    if (P == 1) {
        const float Q = QMAX8, inv = Q / mx;
        const int Qi = 127;
        for (i = 0; i < in; i++) {
            int v = floor_i(x[i] * inv + 0.5f);          // |x * inv| <= 127 (+ rounding)
            v = v > Qi ? Qi : (v < -Qi ? -Qi : v);
            q[i] = (int8_t)v;
        }
        for (i = in; i < ldq; i++) q[i] = 0;             // padding to the 16-byte row stride is ZERO (the S3 kernel runs whole vectors over it)
        return mx / Q;
    }
    const float Q = QMAX16, inv = Q / mx;
    const int Qi = 16256;
    int8_t *hi = q, *lo = q + ldq;
    for (i = 0; i < in; i++) {
        int qi = floor_i(x[i] * inv + 0.5f);             // |x * inv| <= 16256 (+ rounding)
        qi = qi > Qi ? Qi : (qi < -Qi ? -Qi : qi);
        int hh = (qi + 64 + 16384) / 128 - 128;       // floor((qi + 64) / 128)
        hi[i] = (int8_t)hh;
        lo[i] = (int8_t)(qi - 128 * hh);
    }
    for (i = in; i < ldq; i++) { hi[i] = 0; lo[i] = 0; }
    return mx / Q;
}

// Dense int8 / int16-weight conv or linear on n rows. X holds n + K - 1 rows (already zero padded), row stride ldx;
// the first L->in values of each row are the input. Y: n rows, stride ldy.
// Activations: P planes (int16-weight layers always 2, int8-weight layers c->act_planes); weights: WP planes.
// Every (activation plane, weight plane) pair is one exact int8 GEMM; they are combined exactly in integers.
static void qlin_run_(itofs_ctx_t *c, const itofs_qlin_t *L, const float *X, int ldx, int n, float *Y, int ldy, int grp);
static void qlin_run(itofs_ctx_t *c, const itofs_qlin_t *L, const float *X, int ldx, int n, float *Y, int ldy, int grp)
{
    PROF_T(pl0);
    qlin_run_(c, L, X, ldx, n, Y, ldy, grp);
    PROF_ADD(ITOFS_PROF_QLIN, pl0);
}
typedef struct { itofs_ctx_t *c; const float *X; int ldx, in, ldq, P; } quant_arg_t;
static void quant_body(void *a_, int r0, int r1, int core)
{
    const quant_arg_t *a = a_; (void)core;
    for (int r = r0; r < r1; r++)
        a->c->s_qs[r] = quant_row(a->X + (size_t)r * a->ldx, a->in, a->ldq, a->P, a->c->s_q8 + (size_t)r * a->P * a->ldq);
}

typedef struct { const itofs_ctx_t *c; float *Y; int ldy, o0, ot, k, WP, P, exact32; const int32_t *a0, *a1; const itofs_qlin_t *L; int first, last; } resc_arg_t;
static void rescale_body(void *a_, int t0, int t1, int core)
{
    const resc_arg_t *A = a_; (void)core;
    const int ot = A->ot, k = A->k;
    for (int t = t0; t < t1; t++) {
        const float s = A->c->s_qs[t + k];
        if (s == 0.f) continue;
        float *y = A->Y + (size_t)t * A->ldy + A->o0;
        if (A->WP == 1 && A->P == 1) {
            const int32_t *a = A->a0 + (size_t)t * ot;
            for (int o = 0; o < ot; o++) y[o] += s * (float)a[o];
        } else if (A->WP == 1) {
            const int32_t *ah = A->a0 + (size_t)(2 * t) * ot, *al = ah + ot;
            if (A->exact32) for (int o = 0; o < ot; o++) y[o] += s * (float)(128 * ah[o] + al[o]);
            else for (int o = 0; o < ot; o++) y[o] += s * (float)((int64_t)ah[o] * 128 + al[o]);
        } else {                // int16 weights x 15-bit activations: 4 exact products
            const int32_t *hh = A->a0 + (size_t)(2 * t) * ot, *lh = hh + ot;     // (x hi, x lo) x w hi
            const int32_t *hl = A->a1 + (size_t)(2 * t) * ot, *ll = hl + ot;     // (x hi, x lo) x w lo
            for (int o = 0; o < ot; o++) {
                // v = 256 * (128 hh + lh) + (128 hl + ll) = 256 A + B. With in <= 1024 both fit int32 (A->exact32). The float of the exact integer
                // v is taken without 64-bit arithmetic (library calls on the S3) whenever possible: in 32 bits when 256 A + B does not
                // overflow, or as (float)(256 A) + (float)B when both are exact floats (one rounding: the same as rounding v). Else 64 bits.
                float v;
                if (A->exact32) {
                    const int32_t a32 = 128 * hh[o] + lh[o], b32 = 128 * hl[o] + ll[o];
                    if ((uint32_t)a32 + (1u << 23) < (1u << 24)) {                       // |A| < 2^23: 256 A fits int32
                        const int32_t a8 = a32 * 256, v32 = (int32_t)((uint32_t)a8 + (uint32_t)b32);
                        if (((a8 ^ v32) & (b32 ^ v32)) >= 0) { y[o] += s * (float)v32; continue; }
                    } else if ((uint32_t)a32 + (1u << 24) < (1u << 25) && (uint32_t)b32 + (1u << 24) < (1u << 25)) {   // both exact floats
                        y[o] += s * ((float)a32 * 256.f + (float)b32);
                        continue;
                    }
                    v = (float)(((int64_t)a32) * 256 + b32);
                } else {
                    v = (float)(((int64_t)hh[o] * 128 + lh[o]) * 256 + ((int64_t)hl[o] * 128 + ll[o]));
                }
                y[o] += s * v;
            }
        }
    }
}

// One int8 plane x one int8 weight plane (the whole decoder at the default 8-bit activations): tap k of the tile folded into Y in one
// pass. The first tap STORES s*acc (the generic path adds it to the zeroed Y: 0 + v == v, as v is never -0), the last tap applies the
// bias, y * sw + b. Same operations in the same order as rescale_body + bias_body, so the same bits, in one third of the passes.
// (the loops are unrolled by hand: GCC's Xtensa output has no auto-increment addressing, so a plain loop spends more instructions on
// pointers than on arithmetic)
#define RS_LOOP(EXPR) do { int o = 0; for (; o + 4 <= ot; o += 4) { { const int i_ = o;     EXPR; } { const int i_ = o + 1; EXPR; } \
                                                                   { const int i_ = o + 2; EXPR; } { const int i_ = o + 3; EXPR; } } \
                           for (; o < ot; o++) { const int i_ = o; EXPR; } } while (0)
static void rescale1_body(void *a_, int t0, int t1, int core)
{
    const resc_arg_t *A = a_; (void)core;
    const int ot = A->ot, k = A->k, o0 = A->o0;
    const float *restrict sw = A->L->sw + o0, *restrict b = A->L->b ? A->L->b + o0 : NULL;
    for (int t = t0; t < t1; t++) {
        const float s = A->c->s_qs[t + k];
        float *restrict y = A->Y + (size_t)t * A->ldy + o0;
        const int32_t *restrict a = A->a0 + (size_t)t * ot;
        if (A->first && A->last) {
            if (s == 0.f) {                                   // a row of zeros: y = 0 * sw (+ b), as the generic path
                if (b) RS_LOOP(y[i_] = 0.f * sw[i_] + b[i_]); else RS_LOOP(y[i_] = 0.f * sw[i_]);
            } else {
                if (b) RS_LOOP(y[i_] = (s * (float)a[i_]) * sw[i_] + b[i_]); else RS_LOOP(y[i_] = (s * (float)a[i_]) * sw[i_]);
            }
            continue;
        }
        if (A->first) {
            if (s == 0.f) RS_LOOP(y[i_] = 0.f);
            else RS_LOOP(y[i_] = s * (float)a[i_]);
        } else if (s != 0.f) {
            RS_LOOP(y[i_] += s * (float)a[i_]);
        }
        if (A->last) {
            if (b) RS_LOOP(y[i_] = y[i_] * sw[i_] + b[i_]); else RS_LOOP(y[i_] = y[i_] * sw[i_]);
        }
    }
}

typedef struct { const itofs_qlin_t *L; float *Y; int ldy, first; } bias_arg_t;
static void bias_body(void *a_, int t0, int t1, int core)
{
    const bias_arg_t *A = a_; (void)core;
    const itofs_qlin_t *L = A->L;
    const int out = L->out;
    for (int t = t0; t < t1; t++) {
        float *y = A->Y + (size_t)t * A->ldy;
        if (A->first) { memset(y, 0, (size_t)out * 4); continue; }
        if (L->b) for (int o = 0; o < out; o++) y[o] = y[o] * L->sw[o] + L->b[o];
        else for (int o = 0; o < out; o++) y[o] = y[o] * L->sw[o];
    }
}

static void qlin_run_(itofs_ctx_t *c, const itofs_qlin_t *L, const float *X, int ldx, int n, float *Y, int ldy, int grp)
{
    const int in = L->in, out = L->out, K = L->K, ldq = pad16(in);
    const int WP = L->w_lo ? 2 : 1, P = (L->w_lo || L->act16) ? 2 : c->act_planes;
    const int rows = n + K - 1;
    PROF_T(pq0);
    quant_arg_t qa = { c, X, ldx, in, ldq, P };
    par_for(c, quant_body, &qa, rows, (long)rows * in);
    PROF_ADD(ITOFS_PROF_QUANT, pq0);
    const int fast = WP == 1 && P == 1;     // fused store / accumulate / bias epilogue (no zero-fill, no separate bias pass)
    bias_arg_t ba = { L, Y, ldy, 1 };
    if (!fast) for (int t = 0; t < n; t++) memset(Y + (size_t)t * ldy, 0, (size_t)out * 4);
    const int exact32 = in <= 1024;     // |128*acc_hi + acc_lo| <= 16256*127*in < 2^31
    // output channels per GEMM call: as many as the accumulator scratch holds for n * P rows x WP planes (<= ACC_TILE).
    // Each output's arithmetic (taps accumulated in order) does not depend on the tiling.
    int tile = (int)(c->acc_bytes / ((size_t)n * P * WP * 4)) & ~7;
    tile = imin(c->acc_tile, imax(8, tile));
    for (int o0 = 0; o0 < out; o0 += tile) {
        const int ot = imin(tile, out - o0);
        int32_t *a0 = c->s_acc, *a1 = c->s_acc + (size_t)n * P * ot;
        for (int k = 0; k < K; k++) {
            const int8_t *xq = c->s_q8 + (size_t)k * P * ldq;
            const size_t woff = ((size_t)k * out + o0) * in;
            PROF_T(pg0);
            c->qgemm(xq, n * P, ldq, in, L->w + woff, ot, a0, c->qgemm_user);
            if (WP == 2) c->qgemm(xq, n * P, ldq, in, L->w_lo + woff, ot, a1, c->qgemm_user);
            PROF_ADD(ITOFS_PROF_GEMM, pg0);
            resc_arg_t ra = { c, Y, ldy, o0, ot, k, WP, P, exact32, a0, a1, L, k == 0, k == K - 1 };
            PROF_T(pr0);
            par_for(c, fast ? rescale1_body : rescale_body, &ra, n, (long)n * ot * P * WP);
            PROF_ADD(fast ? ITOFS_PROF_P0 + 12 : ITOFS_PROF_P0 + 3, pr0);
        }
    }
    ba.first = 0;
    if (!fast) { PROF_T(pb0); par_for(c, bias_body, &ba, n, (long)n * out); PROF_ADD(ITOFS_PROF_P0 + 4, pb0); }
    c->macs[grp] += (double)n * out * in * K;
    c->macs_exec[grp] += (double)n * out * in * K * P * WP;
}

// qlin over a whole token sequence X[n][ldx] with zero padding (lpad left, K-1-lpad right)
static void qlin_seq(itofs_ctx_t *c, const itofs_qlin_t *L, const float *X, int ldx, int n, int lpad, float *Y, int ldy)
{
    const int in = L->in, K = L->K;
    for (int t0 = 0; t0 < n; t0 += c->tb) {
        const int t1 = imin(n, t0 + c->tb), mm = t1 - t0, rows = mm + K - 1;
        for (int r = 0; r < rows; r++) {
            int t = t0 - lpad + r;
            float *d = c->t_pad + (size_t)r * in;
            if (t < 0 || t >= n) memset(d, 0, (size_t)in * 4);
            else memcpy(d, X + (size_t)t * ldx, (size_t)in * 4);
        }
        qlin_run(c, L, c->t_pad, in, mm, Y + (size_t)t0 * ldy, ldy, ITOFS_G_TEXT);
    }
}

static void layernorm_row(const float *x, int C, const float *g, const float *b, float eps, float *y)
{
    OPC(ITOFS_OPC_LN, C);
    float mu = 0.f;
    for (int i = 0; i < C; i++) mu += x[i];
    mu /= (float)C;
    float var = 0.f;
    for (int i = 0; i < C; i++) { float d = x[i] - mu; var += d * d; }
    var /= (float)C;
    const float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < C; i++) y[i] = (x[i] - mu) * inv * g[i] + b[i];
}

static inline float gelu_v2(float x) { return 0.5f * x * (1.0f + itf_erff(x * 0.70710678118654752f)); }
static inline __attribute__((always_inline)) float gelu_v3(float x) { return 0.5f * x * (1.0f + itf_erff_h(x * 0.70710678118654752f)); }
#define gelu(x) (m_arch3 ? gelu_v3(x) : gelu_v2(x))

// y[o] = b[o] + sum_i W[o][i] x[i]  (f32)
typedef struct { const float *W, *b, *x; float *y; int in; } gemv_arg_t;
static void gemv_body(void *a_, int o0, int o1, int core)
{
    const gemv_arg_t *A = a_; (void)core;
    const int in = A->in;
    for (int o = o0; o < o1; o++) {
        const float *w = A->W + (size_t)o * in;
        float a = 0.f;
        for (int i = 0; i < in; i++) a += w[i] * A->x[i];
        A->y[o] = a + (A->b ? A->b[o] : 0.f);
    }
}
static void gemv_f32(itofs_ctx_t *c, const float *W, const float *b, const float *x, int out, int in, float *y, int grp)
{
    gemv_arg_t a = { W, b, x, y, in };
    par_for(c, gemv_body, &a, out, (long)out * in / 2);
    c->macs_f32[grp] += (double)out * in;
}

// ConvLN over a token sequence x[n][D] (in place): x += gelu(film(LN(conv(x))))
// g1 / bb: the layer's FiLM (1 + g, b) from style_setup(), or NULL (no FiLM)
static void convln_seq(itofs_ctx_t *c, const itofs_convln_t *L, float *x, int n, int K, int rpad, const float *g1, const float *bb)
{
    const int D = L->conv.out, m_arch3 = c->m->arch == 3;
    qlin_seq(c, &L->conv, x, D, n, K - 1 - rpad, c->t_y, D);
    for (int t = 0; t < n; t++) {
        float *y = c->t_y + (size_t)t * D, *xr = x + (size_t)t * D;
        layernorm_row(y, D, L->ln_g, L->ln_b, c->m->eps_text, y);
        if (g1) for (int i = 0; i < D; i++) y[i] = y[i] * g1[i] + bb[i];
        for (int i = 0; i < D; i++) xr[i] = xr[i] + gelu(y[i]);
    }
}

// ======================================================================================================================
// text side
// ======================================================================================================================
static void style_from_vec(itofs_ctx_t *c, const float *sv);

static void style_setup(itofs_ctx_t *c, int style_idx)
{
    const itofs_model_t *m = c->m;
    if (m->style_table) {            // precomputed per style: dur layers, prosody layers, mel layers, each (1 + g, b)
        const float *r = m->style_table + (size_t)style_idx * m->style_table_w;
        const int D = m->text_dim, PD = m->pros_dim, MD = m->mel_dim;
        for (int l = 0; l < m->dur_layers; l++) { memcpy(c->dfilm_g1[l], r, (size_t)D * 4); memcpy(c->dfilm_b[l], r + D, (size_t)D * 4); r += 2 * D; }
        for (int l = 0; l < m->pros_layers; l++) { memcpy(c->film_g1[l], r, (size_t)PD * 4); memcpy(c->film_b[l], r + PD, (size_t)PD * 4); r += 2 * PD; }
        for (int l = 0; l < m->mel_layers; l++) { memcpy(c->mfilm_g1[l], r, (size_t)MD * 4); memcpy(c->mfilm_b[l], r + MD, (size_t)MD * 4); r += 2 * MD; }
        memset(c->cond, 0, (size_t)m->dec_dim * 4);
        return;
    }
    style_from_vec(c, m->style_bank + (size_t)style_idx * m->style_in);
}

// y = W x + b for a per-utterance linear stored f32 (gemv) or with integer weights (one-row GEMM, 15-bit activations)
static void lin1(itofs_ctx_t *c, const float *Wf, const float *bf, const itofs_qlin_t *Q, const float *x, int out, int in, float *y)
{
    if (Q && Q->w) qlin_run(c, Q, x, in, 1, y, out, ITOFS_G_TEXT);
    else gemv_f32(c, Wf, bf, x, out, in, y, ITOFS_G_TEXT);
}

// style vector (style_in) -> s, decoder cond (v2) and every FiLM (1 + g, b)
static void style_from_vec(itofs_ctx_t *c, const float *sv)
{
    const itofs_model_t *m = c->m;
    lin1(c, m->style_w, m->style_b, &m->style_q, sv, m->style_dim, m->style_in, c->s);
    if (m->cond_w) gemv_f32(c, m->cond_w, m->cond_b, c->s, m->dec_dim, m->style_dim, c->cond, ITOFS_G_TEXT);
    else memset(c->cond, 0, (size_t)m->dec_dim * 4);
    for (int l = 0; l < (m->arch == 3 ? m->mel_layers : 0); l++) {
        float *gb = c->t_film;
        const int MD = m->mel_dim;
        lin1(c, m->mel[l].film_w, m->mel[l].film_b, &m->mel[l].film_q, c->s, 2 * MD, m->style_dim, gb);
        for (int i = 0; i < MD; i++) { c->mfilm_g1[l][i] = 1.f + gb[i]; c->mfilm_b[l][i] = gb[MD + i]; }
    }
    for (int l = 0; l < m->pros_layers; l++) {
        float *gb = c->t_film;
        const int PD = m->pros_dim;
        lin1(c, m->pros[l].film_w, m->pros[l].film_b, &m->pros[l].film_q, c->s, 2 * PD, m->style_dim, gb);
        for (int i = 0; i < PD; i++) { c->film_g1[l][i] = 1.f + gb[i]; c->film_b[l][i] = gb[PD + i]; }
    }
    for (int l = 0; l < m->dur_layers; l++) {        // (incremental text side; text_side() computes them itself)
        float *gb = c->t_film;
        const int D = m->text_dim;
        lin1(c, m->dur[l].film_w, m->dur[l].film_b, &m->dur[l].film_q, c->s, 2 * D, m->style_dim, gb);
        for (int i = 0; i < D; i++) { c->dfilm_g1[l][i] = 1.f + gb[i]; c->dfilm_b[l][i] = gb[D + i]; }
    }
}

// text -> style predictor on the whole sentence's encoder output c->h[0..n), then all FiLMs (before any duration)
static void sp_style(itofs_ctx_t *c, int n)
{
    const itofs_model_t *m = c->m;
    const int D = m->text_dim, hid = m->sp[0].out, SI = m->style_in;
    float *f = c->s_g0, *y1 = c->s_y0, *y2 = c->s_g1;
    for (int i = 0; i < D; i++) { f[i] = 0.f; f[D + i] = c->h[i]; }
    for (int t = 0; t < n; t++) {
        const float *h = c->h + (size_t)t * D;
        for (int i = 0; i < D; i++) { f[i] += h[i]; if (h[i] > f[D + i]) f[D + i] = h[i]; }
    }
    for (int i = 0; i < D; i++) f[i] /= (float)n;
    f[2 * D] = itf_logf((float)n) / 5.0f;
    c->macs_f32[ITOFS_G_TEXT] += (double)n * D;
    qlin_run(c, &m->sp[0], f, 2 * D + 1, 1, y1, hid, ITOFS_G_TEXT);
    for (int i = 0; i < hid; i++) y1[i] = gelu_v3(y1[i]);
    qlin_run(c, &m->sp[1], y1, hid, 1, y2, hid, ITOFS_G_TEXT);
    for (int i = 0; i < hid; i++) y2[i] = gelu_v3(y2[i]);
    qlin_run(c, &m->sp[2], y2, hid, 1, y1, SI, ITOFS_G_TEXT);
    for (int i = 0; i < SI; i++) y1[i] += m->sp_mu[i];
    style_from_vec(c, y1);
}

static void text_side(itofs_ctx_t *c, const int *tokens, int n)
{
    const itofs_model_t *m = c->m;
    const int D = m->text_dim, H = m->rnn_hidden;
    float *x = c->t_x;
    for (int t = 0; t < n; t++) memcpy(x + (size_t)t * D, m->emb + (size_t)tokens[t] * D, (size_t)D * 4);
    for (int l = 0; l < m->text_layers; l++) convln_seq(c, &m->enc[l], x, n, m->text_kernel, m->text_rpad, NULL, NULL);
    if (H) {
        float *gh = c->t_film, *hs = gh + 3 * H;
        const int dirs = m->rnn_bidir ? 2 : 1;
        for (int d = 0; d < dirs; d++) {
            qlin_seq(c, &m->rnn_ih[d], x, D, n, 0, c->t_g, 3 * H);
            memset(hs, 0, (size_t)H * 4);
            for (int step = 0; step < n; step++) {
                const int t = d ? n - 1 - step : step;
                gemv_f32(c, m->rnn_hh[d], m->rnn_bhh[d], hs, 3 * H, H, gh, ITOFS_G_TEXT);
                const float *gi = c->t_g + (size_t)t * 3 * H;
                float *o = c->t_o + (size_t)t * dirs * H + d * H;
                for (int j = 0; j < H; j++) {
                    float r = itf_sigmoidf(gi[j] + gh[j]);
                    float z = itf_sigmoidf(gi[H + j] + gh[H + j]);
                    float nn = itf_tanhf(gi[2 * H + j] + r * gh[2 * H + j]);
                    hs[j] = (1.f - z) * nn + z * hs[j];
                    o[j] = hs[j];
                }
            }
        }
        qlin_seq(c, &m->rnn_proj, c->t_o, dirs * H, n, 0, c->t_y, D);
        for (size_t i = 0; i < (size_t)n * D; i++) c->h[i] = x[i] + c->t_y[i];
    } else {
        memcpy(c->h, x, (size_t)n * D * 4);
    }
    if (c->force_h) memcpy(c->h, c->force_h, (size_t)n * D * 4);
    if (m->has_sp) sp_style(c, n);
    memcpy(x, c->h, (size_t)n * D * 4);
    for (int l = 0; l < m->dur_layers; l++) convln_seq(c, &m->dur[l], x, n, m->dur_kernel, m->dur_rpad, c->dfilm_g1[l], c->dfilm_b[l]);
    for (int t = 0; t < n; t++) {
        const float *xr = x + (size_t)t * D;
        float a = 0.f;
        for (int i = 0; i < D; i++) a += m->dur_out_w[i] * xr[i];
        c->logd[t] = a + m->dur_out_b[0];
    }
    c->macs_f32[ITOFS_G_TEXT] += (double)n * D;
}

// torch.round (half to even) of a positive float, clamped to [1, 65535]
static int round_dur(float e)
{
    if (!(e < 65535.f)) return 65535;
    float fl = floorf(e), d = e - fl;
    int i = (int)fl;
    if (d > 0.5f || (d == 0.5f && (i & 1))) i++;
    return i < 1 ? 1 : i;
}

// length-regulated text features of frame t: [h[token], position in token, log(duration)/3 (, t/T)]
static void hf_row(const itofs_ctx_t *c, int t, float *dst)
{
    const itofs_model_t *m = c->m;
    int lo = 0, hi = (c->text_incr ? c->n_done : c->n_tok) - 1;   // tokens whose start is final
    while (lo < hi) { int mid = (lo + hi + 1) / 2; if (c->start[mid] <= t) lo = mid; else hi = mid - 1; }
    const int D = m->text_dim;
    memcpy(dst, c->h + (size_t)lo * D, (size_t)D * 4);
    dst[D] = (float)(t - c->start[lo]) / (float)c->dur[lo];
    dst[D + 1] = itf_logf((float)c->dur[lo]) / 3.0f;
    if (m->sent_pos) dst[D + 2] = (float)t / (float)c->T;
}

// ======================================================================================================================
// incremental text side (forward-only GRU): token stages with ring buffers, exactly as the frame stages
//   ENC_l (conv k, L/R) -> GRU (causal) -> PROJ (h = enc + rnn_proj(o), written to c->h) -> DUR_l (L/R, FiLM) -> DOUT
// Every row is computed with the same primitives and in the same order as the whole-sentence text_side(), so the
// results are bit-identical; only the order in which tokens are computed differs.
// ======================================================================================================================
static inline float *tring_row(const itofs_ctx_t *c, int s, int i)
{
    return c->tst[s].ring + (size_t)(i & c->tring_mask) * c->tst[s].width;
}

// rows [i0, i1) of the input of token stage s (zero outside [0, n)), D floats each, into dst
static void tgather(itofs_ctx_t *c, int s, int i0, int i1, float *dst)
{
    const itofs_model_t *m = c->m;
    const int D = m->text_dim;
    for (int i = i0; i < i1; i++) {
        float *d = dst + (size_t)(i - i0) * D;
        if (i < 0 || i >= c->n_tok) { memset(d, 0, (size_t)D * 4); continue; }
        if (s == 0) memcpy(d, m->emb + (size_t)c->tok[i] * D, (size_t)D * 4);              // embedding
        else if (c->tst[s].kind == TK_DUR && c->tst[s - 1].kind == TK_PROJ) memcpy(d, c->h + (size_t)i * D, (size_t)D * 4);
        else memcpy(d, tring_row(c, s - 1, i), (size_t)D * 4);
    }
}

#ifdef ITOFS_PROF
// instructions per call of the float primitives (profiling build: clock = CCOUNT)
void itofs_prof_micro(double *res)
{
    volatile float sink = 0.f, in = 1.7f, in2 = 0.5f, in3 = 2.0f;
    uint32_t t;
    float acc = 0.f;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += 1.f / (in + (float)i * 1e-4f); res[0] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += gelu_v3(in2 + (float)i * 1e-4f); res[1] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += gelu_v3(in3 + (float)i * 1e-4f); res[2] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += itf_expf(in2 + (float)i * 1e-4f); res[3] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += sqrtf(in + (float)i * 1e-4f); res[4] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) { float a, b; itf_sincosf(in + (float)i * 1e-4f, &a, &b); acc += a + b; } res[5] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += itf_logf(in + (float)i * 1e-4f); res[6] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    t = itofs_prof_clock(); for (int i = 0; i < 1000; i++) acc += acc * in2 + in3; res[7] = (double)(uint32_t)(itofs_prof_clock() - t) / 1000; sink = acc;
    (void)sink;
}
#endif

// ---- row bodies of the frame / token stages (run on one or both cores, par_for) ----
typedef struct { const itofs_ctx_t *c; const itofs_convln_t *L; float *y0; const float *g0; int W, lp; const float *g1, *bb;
                 float *(*dst)(const itofs_ctx_t *, int, int); int s, base, frame; float eps; } cln_arg_t;
static float *tring_dst(const itofs_ctx_t *c, int s, int i) { return tring_row(c, s, i); }
// ConvLN tail of row t: y = LN(conv) (in y0), frame stages: out = x + gelu(y * g1 + b); token stages: FiLM optional,
// out = x + gelu(FiLM(y)) (written to the stage ring for token stages, back into y0 for frame stages)
static void cln_body(void *a_, int t0, int t1, int core)
{
    const cln_arg_t *A = a_; (void)core;
    const int W = A->W, m_arch3 = A->c->m->arch == 3;
    for (int t = t0; t < t1; t++) {
        float *y = A->y0 + (size_t)t * W;
        const float *xr = A->g0 + (size_t)(t + A->lp) * W;
        layernorm_row(y, W, A->L->ln_g, A->L->ln_b, A->eps, y);
        if (A->frame) {
            if (m_arch3) { const float *K = itf_kt; for (int i = 0; i < W; i++) y[i] = xr[i] + gelu_k(y[i] * A->g1[i] + A->bb[i], K); }
            else for (int i = 0; i < W; i++) y[i] = xr[i] + gelu(y[i] * A->g1[i] + A->bb[i]);
        } else {
            float *o = A->dst(A->c, A->s, A->base + t);
            if (A->g1) for (int i = 0; i < W; i++) y[i] = y[i] * A->g1[i] + A->bb[i];
            if (m_arch3) { const float *K = itf_kt; for (int i = 0; i < W; i++) o[i] = xr[i] + gelu_k(y[i], K); }
            else for (int i = 0; i < W; i++) o[i] = xr[i] + gelu(y[i]);
        }
    }
}


static void run_tstage_(itofs_ctx_t *c, int s, int i0, int i1);
static void run_tstage(itofs_ctx_t *c, int s, int i0, int i1)
{
    PROF_T(ps0);
#ifdef ITOFS_PROF
    const double pg = itofs_prof[ITOFS_PROF_GEMM];
#endif
    run_tstage_(c, s, i0, i1);
    PROF_ADD(ITOFS_PROF_TSTAGE + c->tst[s].kind, ps0);
#ifdef ITOFS_PROF
    itofs_prof[ITOFS_PROF_SGEMM + 12 + c->tst[s].kind] += itofs_prof[ITOFS_PROF_GEMM] - pg;
#endif
}
static void run_tstage_(itofs_ctx_t *c, int s, int i0, int i1)
{
    const itofs_model_t *m = c->m;
    const int m_arch3 = m->arch == 3;
    itofs_tstage_t *S = &c->tst[s];
    const int n = i1 - i0, D = m->text_dim, H = m->rnn_hidden;
    switch (S->kind) {
    case TK_ENC: case TK_DUR: {
        const itofs_convln_t *L = S->kind == TK_ENC ? &m->enc[S->idx] : &m->dur[S->idx];
        const float *g1 = S->kind == TK_DUR ? c->dfilm_g1[S->idx] : NULL, *bb = S->kind == TK_DUR ? c->dfilm_b[S->idx] : NULL;
        tgather(c, s, i0 - S->L, i1 + S->R, c->s_g0);
        qlin_run(c, &L->conv, c->s_g0, D, n, c->s_y0, D, ITOFS_G_TEXT);
        cln_arg_t a = { c, L, c->s_y0, c->s_g0, D, S->L, g1, bb, tring_dst, s, i0, 0, m->eps_text };
        par_for(c, cln_body, &a, n, (long)n * D * 8);
        (void)m_arch3;
        break;
    }
    case TK_GRU: {
        float *gh = c->t_film, *hs = c->gru_h;
        tgather(c, s, i0, i1, c->s_g0);
        qlin_run(c, &m->rnn_ih[0], c->s_g0, D, n, c->s_y0, 3 * H, ITOFS_G_TEXT);
        for (int t = 0; t < n; t++) {
            gemv_f32(c, m->rnn_hh[0], m->rnn_bhh[0], hs, 3 * H, H, gh, ITOFS_G_TEXT);
            const float *gi = c->s_y0 + (size_t)t * 3 * H;
            float *o = tring_row(c, s, i0 + t);
            for (int j = 0; j < H; j++) {
                float r = itf_sigmoidf(gi[j] + gh[j]);
                float z = itf_sigmoidf(gi[H + j] + gh[H + j]);
                float nn = itf_tanhf(gi[2 * H + j] + r * gh[2 * H + j]);
                hs[j] = (1.f - z) * nn + z * hs[j];
                o[j] = hs[j];
            }
        }
        break;
    }
    case TK_PROJ: {
        const int se = m->rnn_hidden ? s - 2 : s - 1;   // last encoder stage
        if (m->rnn_hidden) {
            for (int t = 0; t < n; t++) memcpy(c->s_g0 + (size_t)t * H, tring_row(c, s - 1, i0 + t), (size_t)H * 4);
            qlin_run(c, &m->rnn_proj, c->s_g0, H, n, c->s_y0, D, ITOFS_G_TEXT);
        }
        for (int t = 0; t < n; t++) {
            float *h = c->h + (size_t)(i0 + t) * D;
            const float *x = tring_row(c, se, i0 + t);
            if (c->force_h) memcpy(h, c->force_h + (size_t)(i0 + t) * D, (size_t)D * 4);
            else if (m->rnn_hidden) { const float *y = c->s_y0 + (size_t)t * D; for (int i = 0; i < D; i++) h[i] = x[i] + y[i]; }
            else memcpy(h, x, (size_t)D * 4);
        }
        break;
    }
    case TK_DOUT: {
        for (int t = 0; t < n; t++) {
            const int i = i0 + t;
            const float *xr = tring_row(c, s - 1, i);
            float a = 0.f;
            for (int k = 0; k < D; k++) a += m->dur_out_w[k] * xr[k];
            c->logd[i] = a + m->dur_out_b[0];
            const int d = c->force_dur ? imax(1, c->force_dur[i]) : round_dur(itf_expf(c->logd[i]));
            c->dur[i] = d;
            c->start[i + 1] = c->start[i] + d;
        }
        c->macs_f32[ITOFS_G_TEXT] += (double)n * D;
        c->n_done = i1;
        if (c->n_done == c->n_tok) { c->T = c->start[c->n_tok] < 1 ? 1 : c->start[c->n_tok]; c->T_known = 1; }
        break;
    }
    }
}

// compute the token stages so that the first `target` tokens have final durations
static void text_advance_to(itofs_ctx_t *c, int target, int S)
{
    const int n = c->n_tok;
    int need[MAX_TSTAGES];
    need[S - 1] = imin(target, n);
    for (int s = S - 1; s >= 1; s--) need[s - 1] = imin(n, need[s] + c->tst[s].R);
    for (int s = 0; s < S; s++) {
        itofs_tstage_t *st = &c->tst[s];
        while (st->done < need[s]) {
            const int i1 = imin(need[s], st->done + st->bmax);
            run_tstage(c, s, st->done, i1);
            st->done = i1;
        }
    }
}

static void text_advance(itofs_ctx_t *c, int target) { text_advance_to(c, target, c->n_tst); }

// style predictor on the incremental text side: encoder + GRU + rnn_proj (up to TK_PROJ) for the whole sentence first
static void text_h_all(itofs_ctx_t *c)
{
    int sp = 0;
    while (sp < c->n_tst && c->tst[sp].kind != TK_PROJ) sp++;
    for (int t = 0; t < c->n_tok; ) { t = imin(c->n_tok, t + c->nb); text_advance_to(c, t, sp + 1); }
}

// make the length-regulated features final for frames [0, F) (or finish the text side). Tokens are added in steps of
// nb (the chunk size), so the int GEMMs keep their weight reuse across rows.
static void text_ensure(itofs_ctx_t *c, int F)
{
    // the first chunk adds only FIRST_TEXT_STEP tokens (least work before the first audio); later chunks add nb, so the text-side
    // int16 weights (0.9 MB) are read once per nb tokens
    const int step = c->text_step > 0 ? c->text_step : (c->produced == 0 ? imin(c->nb, FIRST_TEXT_STEP) : c->nb);
    while (c->text_incr && !c->T_known && c->start[c->n_done] < F) text_advance(c, c->n_done + step);
}

// ======================================================================================================================
// source noise: counter-based, so sample n's value does not depend on chunking
// ======================================================================================================================
static uint32_t mix32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// the Gaussian noise of samples 2 * pair (cosine half) and 2 * pair + 1 (sine half) of one Box-Muller pair: one log, sqrt, sincos for both
static void gauss_pair(uint32_t seed, uint32_t pair, float *even, float *odd)
{
    OPC(ITOFS_OPC_GAUSS, 2);
    const uint32_t a = mix32(seed ^ mix32(2u * pair + 0x9e3779b9u)), b = mix32(seed ^ mix32(2u * pair + 1u + 0x9e3779b9u) ^ 0x85ebca6bu);
    const float u1 = ((float)(a >> 8) + 0.5f) * (1.0f / 16777216.0f), u2 = ((float)(b >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float r = sqrtf(-2.f * itf_logf(u1));
    float s, co;
    itf_sincosf_k(6.28318530717958648f * u2, &s, &co, itf_kt);
    *even = r * co; *odd = r * s;
}

// ======================================================================================================================
// stage runners
// ======================================================================================================================
typedef struct { itofs_ctx_t *c; int s, t0; } st_arg_t;
static inline float *ring_row(const itofs_ctx_t *c, int s, int t)
{
    return c->st[s].ring + (size_t)(t & (c->st[s].rframes - 1)) * c->st[s].width;
}

// rows [t0, t1) of producer p (w floats from column off), zero outside [0, T)
static void gather(itofs_ctx_t *c, int p, int t0, int t1, int off, int w, float *dst, int ldd)
{
    PROF_T(pg0);
    for (int t = t0; t < t1; t++) {
        float *d = dst + (size_t)(t - t0) * ldd;
        if (t < 0 || t >= c->T) memset(d, 0, (size_t)w * 4);
        else memcpy(d, ring_row(c, p, t) + off, (size_t)w * 4);
    }
    PROF_ADD(ITOFS_PROF_P0 + 5, pg0);
}

static void put_rows(itofs_ctx_t *c, int s, int t0, int n, const float *src, int lds)
{
    const int w = c->st[s].width;
    PROF_T(pp0);
    for (int t = 0; t < n; t++) memcpy(ring_row(c, s, t0 + t), src + (size_t)t * lds, (size_t)w * 4);
    PROF_ADD(ITOFS_PROF_P0 + 5, pp0);
}

// one piece of the source: samples n0 .. n0+len-1 whose F0 is interpolated between a (at w=0) and b (at w=1) with
// w_j = (j + 0.5) / hop, j = j0 .. j0+len-1 (constant: f = a). Phase anchored in double at the piece start (P).
static void src_piece(const itofs_ctx_t *c, double P, int n0, int len, float a, float b, int j0, int constant, float *out)
{
    OPC(ITOFS_OPC_SRC, len);
    const itofs_model_t *m = c->m;
    const float hop = (float)m->hop, sr = (float)m->sr, half_sr = 0.5f * (float)m->sr;
    const float d = constant ? 0.f : (b - a) / hop;
    const float Pf = (float)(P - floor(P));
    const float f_first = constant ? a : a + d * ((float)j0 + 0.5f);
    const float amp_uv = m->src_amp / 3.0f;
    const int pow2_nh = (m->n_harm & (m->n_harm - 1)) == 0;          // x / 2^k == x * 2^-k exactly
    const float inv_nh = 1.f / (float)m->n_harm;
    uint32_t gpair = 0xffffffffu;       // the Box-Muller pair held in (g_e, g_o); each pair serves two consecutive samples
    float g_e = 0.f, g_o = 0.f;
    for (int J = 0; J < len; J++) {
        const float j = (float)(j0 + J);
        float fl, f;
        if (constant) { fl = a; f = a; }
        else {
            const float w = c->src_w[j0 + J];           // (j + 0.5) / hop, tabulated at init (the same division)
            fl = a + d * (j + 0.5f);                 // same line, closed form for the phase sum
            f = a * (1.f - w) + b * w;               // torch's form for the voicing / Nyquist tests
        }
        const int n = n0 + J;
        float nz;
        if (c->ext_noise) nz = c->ext_noise[n];
        else {
            if ((uint32_t)n >> 1 != gpair) { gpair = (uint32_t)n >> 1; gauss_pair(c->seed, gpair, &g_e, &g_o); }
            nz = (n & 1) ? g_o : g_e;
        }
        if (!(f > m->uv_hz)) { out[J] = amp_uv * nz; continue; }       // unvoiced: the phase and harmonics are not used
        // phase after this sample: P + sum_{m<=J} f_m / sr (arithmetic series)
        float phi = Pf + ((float)(J + 1) * (f_first + fl) * 0.5f) / sr;
        phi = phi - itf_floorf(phi);
        float s1, c1;
        itf_sincosf_k(6.28318530717958648f * phi, &s1, &c1, itf_kt);
        float sk = s1, ck = c1, hsum = 0.f;
        for (int k = 1; k <= m->n_harm; k++) {
            if ((float)k * f < half_sr) hsum += sk;
            const float ns = sk * c1 + ck * s1, nc = ck * c1 - sk * s1;
            sk = ns; ck = nc;
        }
        out[J] = m->src_amp * (pow2_nh ? hsum * inv_nh : hsum / (float)m->n_harm) + m->src_noise * nz;
    }
}

// the double phase anchor after a piece: P + the exact sum of its frequencies / sr
static double src_advance(const itofs_ctx_t *c, double P, int len, float a, float b, int j0, int constant)
{
    const itofs_model_t *m = c->m;
    const double da = a, db = b, dd = constant ? 0.0 : (db - da) / (double)m->hop;
    const double f0d = constant ? da : da + dd * ((double)j0 + 0.5), f1d = constant ? da : da + dd * ((double)(j0 + len - 1) + 0.5);
    return P + (double)len * (f0d + f1d) * 0.5 / (double)m->sr;
}

// frame t's two pieces: A = samples [t*hop, t*hop + hop/2) -> segment t-1 (constant f0[0] before the first frame
// centre); B = [t*hop + hop/2, (t+1)*hop) -> segment t (constant f0[T-1] after the last frame centre)
typedef struct { int n0, len, j0, constant; float a, b; } src_pc_t;
static void src_pieces(const itofs_ctx_t *c, int s, int t, src_pc_t *pc)
{
    const int hop = c->m->hop, T = c->T, h2 = hop / 2, p = c->st[s].prod[0];
    if (t == 0) pc[0] = (src_pc_t){ 0, h2, 0, 1, ring_row(c, p, 0)[3], 0.f };
    else pc[0] = (src_pc_t){ t * hop, h2, h2, 0, ring_row(c, p, t - 1)[3], ring_row(c, p, t)[3] };
    if (t == T - 1) pc[1] = (src_pc_t){ t * hop + h2, hop - h2, 0, 1, ring_row(c, p, t)[3], 0.f };
    else pc[1] = (src_pc_t){ t * hop + h2, hop - h2, 0, 0, ring_row(c, p, t)[3], ring_row(c, p, t + 1)[3] };
}

// phase anchors of frame t's pieces (in order: advances c->src_ph)
static void src_frame_anchors(itofs_ctx_t *c, int s, int t, double *anc)
{
    src_pc_t pc[2];
    src_pieces(c, s, t, pc);
    for (int k = 0; k < 2; k++) {
        anc[k] = c->src_ph;
        c->src_ph = src_advance(c, c->src_ph, pc[k].len, pc[k].a, pc[k].b, pc[k].j0, pc[k].constant);
    }
}

static void src_body(void *a_, int i0, int i1, int core)
{
    const st_arg_t *A = a_; (void)core;
    const itofs_ctx_t *c = A->c;
    const int h2 = c->m->hop / 2;
    for (int i = i0; i < i1; i++) {
        const int t = A->t0 + i;
        src_pc_t pc[2];
        src_pieces(c, A->s, t, pc);
        float *out = ring_row(c, A->s, t);
        for (int k = 0; k < 2; k++)
            src_piece(c, c->src_anc[2 * i + k], pc[k].n0, pc[k].len, pc[k].a, pc[k].b, pc[k].j0, pc[k].constant, out + (k ? h2 : 0));
    }
}

// forward real FFT of N windowed samples -> row [re(0..M), im(0..M)] (core: whose FFT scratch)
static void rfft_row(itofs_ctx_t *c, const float *x, float *row, int core)
{
    const int N = c->m->n_fft, M = N / 2;
    cpx *z = (cpx *)(core ? c->fbuf_b : c->fbuf);
    for (int n = 0; n < M; n++) { z[n].r = x[2 * n]; z[n].i = x[2 * n + 1]; }
    fft_fwd(core ? c->fft_b : c->fft, z);
    float *re = row, *im = row + M + 1;
    re[0] = z[0].r + z[0].i; im[0] = 0.f;
    re[M] = z[0].r - z[0].i; im[M] = 0.f;
    for (int k = 1; k < M; k++) {
        const cpx a = z[k], b = z[M - k];
        const float er = 0.5f * (a.r + b.r), ei = 0.5f * (a.i - b.i);          // E = (Z[k] + conj(Z[M-k])) / 2
        const float orr = 0.5f * (a.i + b.i), oi = -0.5f * (a.r - b.r);       // O = (Z[k] - conj(Z[M-k])) / (2i)
        const float cs = c->rtw[2 * k], sn = c->rtw[2 * k + 1];              // W^k = cs - i sn
        re[k] = er + (orr * cs + oi * sn);
        im[k] = ei + (oi * cs - orr * sn);
    }
}

// inverse real FFT of the half spectrum X (interleaved re, im; M+1 bins) times the window -> out[N]
static void irfft_win(itofs_ctx_t *c, const float *X, float *out, int core)
{
    const int N = c->m->n_fft, M = N / 2;
    cpx *buf = (cpx *)(core ? c->fbuf_b : c->fbuf);
    for (int k = 0; k < M; k++) {
        float ar = X[2 * k], ai = X[2 * k + 1];
        float br = X[2 * (M - k)], bi = -X[2 * (M - k) + 1];
        float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
        float dr = 0.5f * (ar - br), di = 0.5f * (ai - bi);
        float cs = c->rtw[2 * k], sn = c->rtw[2 * k + 1];
        float orr = dr * cs - di * sn, oi = dr * sn + di * cs;
        buf[k].r = er - oi;
        buf[k].i = -(ei + orr);
    }
    fft_fwd(core ? c->fft_b : c->fft, buf);
    const float invM = 1.0f / (float)M;
    for (int j = 0; j < M; j++) {
        out[2 * j] = buf[j].r * invM * c->win[2 * j];
        out[2 * j + 1] = -buf[j].i * invM * c->win[2 * j + 1];
    }
}

typedef struct { itofs_ctx_t *c; const itofs_block_t *bk; int W, K, DI, lp; } blk_arg_t;
static void blk_dw_body(void *a_, int t0, int t1, int core)         // depthwise conv + LayerNorm of rows [t0, t1)
{
    const blk_arg_t *A = a_; (void)core;
    const itofs_ctx_t *c = A->c;
    const int W = A->W, K = A->K;
    for (int t = t0; t < t1; t++) {
        float *y = c->s_y0 + (size_t)t * W;
        for (int ch = 0; ch < W; ch++) {
            float a = 0.f;
            const float *w = A->bk->dw_w + (size_t)ch * K;
            for (int k = 0; k < K; k++) a += c->s_g0[(size_t)(t + k) * W + ch] * w[k];
            y[ch] = a + A->bk->dw_b[ch];
        }
        layernorm_row(y, W, A->bk->ln_g, A->bk->ln_b, c->m->eps_dec, y);
    }
}
static void blk_gelu_body(void *a_, int t0, int t1, int core)
{
    const blk_arg_t *A = a_; (void)core;
    const int m_arch3 = A->c->m->arch == 3;
    float *y1 = A->c->s_y1;
    if (m_arch3) {
        const float *K = itf_kt;
        for (size_t j = (size_t)t0 * A->DI; j < (size_t)t1 * A->DI; j++) y1[j] = gelu_k(y1[j], K);
        return;
    }
    for (size_t j = (size_t)t0 * A->DI; j < (size_t)t1 * A->DI; j++) y1[j] = gelu(y1[j]);
}
static void blk_res_body(void *a_, int t0, int t1, int core)
{
    const blk_arg_t *A = a_; (void)core;
    const itofs_ctx_t *c = A->c;
    const int W = A->W;
    for (int t = t0; t < t1; t++) {
        float *y = c->s_y0 + (size_t)t * W;
        const float *r = c->s_g0 + (size_t)(t + A->lp) * W;
        for (int ch = 0; ch < W; ch++) y[ch] = r[ch] + y[ch] * A->bk->gamma[ch];
    }
}

static void hft_body(void *a_, int i0, int i1, int core)            // reflect-padded windowed source frame -> rFFT row
{
    const st_arg_t *A = a_;
    itofs_ctx_t *c = A->c;
    const itofs_model_t *m = c->m;
    const int hop = m->hop, N = m->n_fft, T = c->T, p = c->st[A->s].prod[0];
    for (int i = i0; i < i1; i++) {
        const int t = A->t0 + i;
        float *x = core ? c->x_b : c->s_y1;     // N windowed samples
        for (int j = 0; j < N; j++) {
            int zi = t * hop + j - N / 2;
            if (zi < 0) zi = -zi;       // reflect pad (center=True)
            const int si = zi - hop / 2; // hop/2 zeros each side of the source
            const float v = (si < 0 || si >= T * hop) ? 0.f : ring_row(c, p, si / hop)[si % hop];
            x[j] = v * c->win[j];
        }
        float *row = c->s_y0 + (size_t)i * (N + 2);
        rfft_row(c, x, row, core);
        if (c->tap) c->tap(c->tap_user, ITOFS_TAP_HFEAT, t, row, N + 2);
    }
}
static void head_body(void *a_, int i0, int i1, int core)           // log-mag / phase -> windowed iSTFT frame
{
    const st_arg_t *A = a_;
    itofs_ctx_t *c = A->c;
    const itofs_model_t *m = c->m;
    const int N = m->n_fft, NB = N / 2 + 1;
    for (int i = i0; i < i1; i++) {
        const float *h = c->s_y0 + (size_t)i * (N + 2);
        if (c->tap) c->tap(c->tap_user, ITOFS_TAP_SPEC, A->t0 + i, h, N + 2);
        float *X = core ? c->x_b : c->s_y1;     // interleaved half spectrum
        const float *K = itf_kt, mag_max = m->mag_max;
        for (int k = 0; k < NB; k++) {
            float mag = itf_expf_k(h[k], K);
            if (mag > mag_max) mag = mag_max;
            float sn, cs;
            itf_sincosf_k(h[NB + k], &sn, &cs, K);
            X[2 * k] = mag * cs;
            X[2 * k + 1] = mag * sn;
        }
        X[1] = 0.f; X[2 * (NB - 1) + 1] = 0.f;    // c2r ignores imag(DC), imag(Nyquist)
        irfft_win(c, X, ring_row(c, A->s, A->t0 + i), core);
    }
}
static void ola_body(void *a_, int i0, int i1, int core)
{
    const st_arg_t *A = a_; (void)core;
    itofs_ctx_t *c = A->c;
    const itofs_stage_t *S = &c->st[A->s];
    const int hop = c->m->hop, N = c->m->n_fft, p = S->prod[0];
    for (int t = A->t0 + i0; t < A->t0 + i1; t++) {
        float *out = c->out_ptr + (size_t)(t - c->out_base) * hop;
        const int f0 = imax(0, t - S->L[0]), f1 = imin(c->T - 1, t + S->R[0]);
        for (int sx = 0; sx < hop; sx++) {
            const int pp = t * hop + sx + c->ola_off;
            float acc = 0.f, env = 0.f;
            for (int f = f0; f <= f1; f++) {
                const int k = pp - f * hop;
                if (k < 0 || k >= N) continue;
                acc += ring_row(c, p, f)[k];
                env += c->win2[k];
            }
            out[sx] = acc / env;
        }
    }
}
typedef struct { itofs_ctx_t *c; int s, ph, t0, W; } emb_arg_t;
static void emb_ln_body(void *a_, int i0, int i1, int core)       // decoder embed: (+ cond) -> LayerNorm -> + text-side harmonic features
{
    const emb_arg_t *A = a_; (void)core;
    const itofs_ctx_t *c = A->c;
    const itofs_model_t *m = c->m;
    const int W = A->W;
    for (int t = i0; t < i1; t++) {
        float *y = c->s_y0 + (size_t)t * W;
        if (m->arch == 2) for (int i = 0; i < W; i++) y[i] += c->cond[i];
        layernorm_row(y, W, m->n0_g, m->n0_b, m->eps_dec, y);
        const float *hp = ring_row(c, A->ph, A->t0 + t);
        for (int i = 0; i < W; i++) y[i] += hp[i];
    }
}
typedef struct { itofs_ctx_t *c; int D; } head_ln_arg_t;
static void head_ln_body(void *a_, int i0, int i1, int core)      // final LayerNorm of the decoder rows (in place)
{
    const head_ln_arg_t *A = a_; (void)core;
    const itofs_ctx_t *c = A->c;
    const itofs_model_t *m = c->m;
    for (int t = i0; t < i1; t++) layernorm_row(c->s_g0 + (size_t)t * A->D, A->D, m->n1_g, m->n1_b, m->eps_dec, c->s_g0 + (size_t)t * A->D);
}

static void run_stage_(itofs_ctx_t *c, int s, int t0, int t1);
static void run_stage(itofs_ctx_t *c, int s, int t0, int t1)
{
    PROF_T(ps0);
#ifdef ITOFS_PROF
    const double pg = itofs_prof[ITOFS_PROF_GEMM];
#endif
    run_stage_(c, s, t0, t1);
    PROF_ADD(ITOFS_PROF_STAGE + c->st[s].kind, ps0);
#ifdef ITOFS_PROF
    itofs_prof[ITOFS_PROF_SGEMM + c->st[s].kind] += itofs_prof[ITOFS_PROF_GEMM] - pg;
#endif
}
static void run_stage_(itofs_ctx_t *c, int s, int t0, int t1)
{
    const itofs_model_t *m = c->m;
    itofs_stage_t *S = &c->st[s];
    const int n = t1 - t0, W = S->width;
    switch (S->kind) {
    case SK_PIN: {
        for (int t = t0; t < t1; t++) hf_row(c, t, c->s_g0 + (size_t)(t - t0) * m->hf_dim);
        qlin_run(c, &m->pros_in, c->s_g0, m->hf_dim, n, c->s_y0, W, ITOFS_G_PROS);
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_PROS: case SK_MEL: {
        const int mel = S->kind == SK_MEL;
        const itofs_convln_t *L = mel ? &m->mel[S->idx] : &m->pros[S->idx];
        const int p = S->prod[0], lp = S->L[0];
        gather(c, p, t0 - lp, t1 + S->R[0], 0, W, c->s_g0, W);
        qlin_run(c, &L->conv, c->s_g0, W, n, c->s_y0, W, mel ? ITOFS_G_MEL : ITOFS_G_PROS);
        const float *g1 = mel ? c->mfilm_g1[S->idx] : c->film_g1[S->idx], *bb = mel ? c->mfilm_b[S->idx] : c->film_b[S->idx];
        cln_arg_t a = { c, L, c->s_y0, c->s_g0, W, lp, g1, bb, NULL, s, t0, 1, m->eps_text };
        PROF_T(pc0);
        par_for(c, cln_body, &a, n, (long)n * W * 8);
        PROF_ADD(ITOFS_PROF_P0 + 11, pc0);
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_CUR: {
        const int PD = m->pros_dim;
        gather(c, S->prod[0], t0, t1, 0, PD, c->s_g0, PD);
        for (int t = 0; t < n; t++) {
            float pred[3];
            if (c->force_pred) memcpy(pred, c->force_pred + (size_t)(t0 + t) * 3, sizeof pred);
            else for (int j = 0; j < 3; j++) {
                const float *w = m->pros_out_w + (size_t)j * PD, *x = c->s_g0 + (size_t)t * PD;
                float a = 0.f;
                for (int i = 0; i < PD; i++) a += w[i] * x[i];
                pred[j] = a + m->pros_out_b[j];
            }
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_PRED, t0 + t, pred, 3);
            float *r = ring_row(c, s, t0 + t);
            const int v = pred[1] > 0.f;
            r[0] = v ? pred[0] : 0.f;
            r[1] = v ? 1.f : 0.f;
            r[2] = pred[2];
            r[3] = v ? itf_expf(pred[0] * m->stats[1] + m->stats[0]) : 0.f;
            if (m->arch == 3) {
                // Vocoder.clean_f0 (F0 <= voc_hz -> unvoiced) and Vocoder.feats: [lf0n_voc, voiced_voc]
                if (!(r[3] > m->voc_hz)) r[3] = 0.f;
                if (c->force_f0) r[3] = c->force_f0[t0 + t];
                const int vv = r[3] > m->voc_hz;
                const float f0c = r[3] < 1.f ? 1.f : r[3];
                r[4] = vv ? (itf_logf(f0c) - m->f0stats[0]) / m->f0stats[1] : 0.f;
                r[5] = vv ? 1.f : 0.f;
            } else if (c->force_f0) r[3] = c->force_f0[t0 + t];
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_F0, t0 + t, &r[3], 1);
        }
        c->macs_f32[ITOFS_G_PROS] += (double)n * 3 * PD;
        break;
    }
    case SK_SRC: {
        // the phase anchors chain from frame to frame (cheap, in order); the samples are then independent per frame
        PROF_T(ps0);
        for (int t = t0; t < t1; t++) src_frame_anchors(c, s, t, c->src_anc + 2 * (t - t0));
        st_arg_t a = { c, s, t0 };
        par_for(c, src_body, &a, n, (long)n * W * 16);
        PROF_ADD(ITOFS_PROF_P0 + 9, ps0);
        for (int t = t0; t < t1; t++) if (c->tap) c->tap(c->tap_user, ITOFS_TAP_SRC, t, ring_row(c, s, t), W);
        break;
    }
    case SK_HFT: {
        const int N = m->n_fft;
        st_arg_t a = { c, s, t0 };
        { PROF_T(pf0); par_for(c, hft_body, &a, n, (long)n * N * 8); PROF_ADD(ITOFS_PROF_P0 + 8, pf0); }
        qlin_run(c, &m->harm_proj, c->s_y0, N + 2, n, c->s_y0, W, ITOFS_G_HARM);     // the wide input is dead once quantised: the output reuses y0
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_MIN: {
        const int pc = S->prod[0], HD = m->hf_dim, DI = HD + 3;
        for (int t = t0; t < t1; t++) {
            float *d = c->s_g0 + (size_t)(t - t0) * DI;
            hf_row(c, t, d);
            const float *cr = ring_row(c, pc, t);
            d[HD] = cr[0]; d[HD + 1] = cr[1]; d[HD + 2] = cr[2];
        }
        qlin_run(c, &m->mel_in, c->s_g0, DI, n, c->s_y0, W, ITOFS_G_MEL);
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_MOUT: {
        const int pm = S->prod[0], pc = S->prod[1], MD = m->mel_dim, NM = m->n_mels;
        gather(c, pm, t0, t1, 0, MD, c->s_g0, MD);
        qlin_run(c, &m->mel_out, c->s_g0, MD, n, c->s_y0, NM, ITOFS_G_MEL);
        for (int t = 0; t < n; t++) {
            float *r = ring_row(c, s, t0 + t);
            const float *y = c->s_y0 + (size_t)t * NM, *cr = ring_row(c, pc, t0 + t);
            if (c->force_mel) memcpy(r, c->force_mel + (size_t)(t0 + t) * NM, (size_t)NM * 4);
            else for (int i = 0; i < NM; i++) r[i] = y[i] * m->mel_std[i] + m->mel_mean[i];
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_MEL, t0 + t, r, NM);
            r[NM] = cr[4]; r[NM + 1] = cr[5];
        }
        break;
    }
    case SK_EMB: {
        const int pc = S->prod[0], ph = S->prod[1], lp = S->L[0], rp = S->R[0], HD = m->hf_dim, DI = m->dec_in;
        if (m->arch == 3) gather(c, pc, t0 - lp, t1 + rp, 0, DI, c->s_g0, DI);   // [log-mel | lf0n_voc | voiced_voc]
        else for (int t = t0 - lp; t < t1 + rp; t++) {
            float *d = c->s_g0 + (size_t)(t - t0 + lp) * DI;
            if (t < 0 || t >= c->T) { memset(d, 0, (size_t)DI * 4); continue; }
            hf_row(c, t, d);
            const float *cr = ring_row(c, pc, t);
            d[HD] = cr[0]; d[HD + 1] = cr[1]; d[HD + 2] = cr[2];
        }
        qlin_run(c, &m->dec_embed, c->s_g0, DI, n, c->s_y0, W, ITOFS_G_EMBED);
        { PROF_T(pl0); emb_arg_t ea = { c, s, ph, t0, W }; par_for(c, emb_ln_body, &ea, n, (long)n * W * 8); PROF_ADD(ITOFS_PROF_P0 + 6, pl0); }
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_BLK: {
        const itofs_block_t *bk = &m->blk[S->idx];
        const int K = m->dec_kernel, lp = S->L[0], DI = m->dec_inter;
        gather(c, S->prod[0], t0 - lp, t1 + S->R[0], 0, W, c->s_g0, W);
        blk_arg_t a = { c, bk, W, K, DI, lp };
        { PROF_T(pp); par_for(c, blk_dw_body, &a, n, (long)n * W * K); PROF_ADD(ITOFS_PROF_P0 + 0, pp); }
        c->macs_f32[ITOFS_G_BLOCKS] += (double)n * W * K;
        qlin_run(c, &bk->pw1, c->s_y0, W, n, c->s_y1, DI, ITOFS_G_BLOCKS);
        { PROF_T(pp); par_for(c, blk_gelu_body, &a, n, (long)n * DI * 8); PROF_ADD(ITOFS_PROF_P0 + 1, pp); }
        qlin_run(c, &bk->pw2, c->s_y1, DI, n, c->s_y0, W, ITOFS_G_BLOCKS);    // y0 (the dw output) is dead once pw1 has quantised it
        { PROF_T(pp); par_for(c, blk_res_body, &a, n, (long)n * W); PROF_ADD(ITOFS_PROF_P0 + 2, pp); }
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_HEAD: {
        const int D = m->dec_dim, N = m->n_fft, NB = N / 2 + 1;
        gather(c, S->prod[0], t0, t1, 0, D, c->s_g0, D);
        { PROF_T(pl0); head_ln_arg_t ha = { c, D }; par_for(c, head_ln_body, &ha, n, (long)n * D * 8); PROF_ADD(ITOFS_PROF_P0 + 6, pl0); }
        qlin_run(c, &m->dec_out, c->s_g0, D, n, c->s_y0, N + 2, ITOFS_G_HEAD);
        st_arg_t a = { c, s, t0 };
        PROF_T(ph0);
        par_for(c, head_body, &a, n, (long)n * NB * 8);
        PROF_ADD(ITOFS_PROF_P0 + 7, ph0);
        break;
    }
    case SK_OLA: {
        st_arg_t a = { c, s, t0 };
        PROF_T(po0);
        par_for(c, ola_body, &a, n, (long)n * m->hop * 4);
        PROF_ADD(ITOFS_PROF_P0 + 10, po0);
        break;
    }
    }
}

// Largest batch (rows) that stage / token stage can run with the hot scratch as allocated: every buffer the stage uses
// must hold n (+ conv halo) rows. At most MAX_FILL; the wide layers (harm_proj, head output) fit only WIDE_ROWS.
static int fits(const itofs_ctx_t *c, int n, int halo, long g0w, long y0w, long y1w, long y2w, int qin, int P, int WP)
{
    const int r = n + halo;
    return (long)r * g0w <= c->cap_g0 && (long)n * (y0w > y2w ? y0w : y2w) <= c->cap_y0 && (long)n * y1w <= c->cap_y1 &&
           r <= c->cap_qs && (long)r * P * pad16(qin) <= c->cap_q8 && (size_t)n * P * WP * 4 * 8 <= c->acc_bytes;
}
static int qP(const itofs_ctx_t *c, const itofs_qlin_t *q) { return (q->w_lo || q->act16) ? 2 : c->act_planes; }
static int qWP(const itofs_qlin_t *q) { return q->w_lo ? 2 : 1; }
static int stage_fits(const itofs_ctx_t *c, const itofs_stage_t *S, int n)
{
    const itofs_model_t *m = c->m;
    const int W = S->width, N = m->n_fft;
    switch (S->kind) {
    case SK_PIN: return fits(c, n, 0, m->hf_dim, W, 0, 0, m->hf_dim, qP(c, &m->pros_in), qWP(&m->pros_in));
    case SK_PROS: case SK_MEL: {
        const itofs_qlin_t *q = S->kind == SK_MEL ? &m->mel[S->idx].conv : &m->pros[S->idx].conv;
        return fits(c, n, q->K - 1, W, W, 0, 0, W, qP(c, q), qWP(q));
    }
    case SK_CUR: return fits(c, n, 0, m->pros_dim, 0, 0, 0, 0, 1, 1);
    case SK_SRC: return n <= imax(c->nb, MAX_FILL);
    case SK_HFT: return fits(c, n, 0, 0, N + 2, 0, W, N + 2, qP(c, &m->harm_proj), qWP(&m->harm_proj));
    case SK_MIN: return fits(c, n, 0, m->hf_dim + 3, W, 0, 0, m->hf_dim + 3, qP(c, &m->mel_in), qWP(&m->mel_in));
    case SK_MOUT: return fits(c, n, 0, m->mel_dim, m->n_mels, 0, 0, m->mel_dim, qP(c, &m->mel_out), qWP(&m->mel_out));
    case SK_EMB: return fits(c, n, m->embed_kernel - 1, m->dec_in, W, 0, 0, m->dec_in, qP(c, &m->dec_embed), qWP(&m->dec_embed));
    case SK_BLK: {
        const itofs_block_t *bk = &m->blk[S->idx];
        return fits(c, n, m->dec_kernel - 1, W, W, m->dec_inter, W, m->dec_inter, imax(qP(c, &bk->pw1), qP(c, &bk->pw2)), 1) &&
               fits(c, n, 0, 0, 0, 0, 0, W, qP(c, &bk->pw1), 1);
    }
    case SK_HEAD: return fits(c, n, 0, m->dec_dim, N + 2, 0, 0, m->dec_dim, qP(c, &m->dec_out), qWP(&m->dec_out));
    default: return 1;      // OLA: writes the caller's rows only
    }
}
static int tstage_fits(const itofs_ctx_t *c, const itofs_tstage_t *S, int n)
{
    const itofs_model_t *m = c->m;
    const int D = m->text_dim, H = m->rnn_hidden;
    switch (S->kind) {
    case TK_ENC: case TK_DUR: {
        const itofs_qlin_t *q = S->kind == TK_ENC ? &m->enc[S->idx].conv : &m->dur[S->idx].conv;
        return fits(c, n, S->L + S->R, D, D, 0, 0, D, qP(c, q), qWP(q));
    }
    case TK_GRU: return fits(c, n, 0, D, 3 * H, 0, 0, D, qP(c, &m->rnn_ih[0]), qWP(&m->rnn_ih[0]));
    case TK_PROJ: return !H || fits(c, n, 0, H, D, 0, 0, H, qP(c, &m->rnn_proj), qWP(&m->rnn_proj));
    default: return 1;
    }
}
static int set_bmax(itofs_ctx_t *c)
{
    for (int s = 0; s < c->n_stages; s++) {
        int b = 1;
        if (!stage_fits(c, &c->st[s], 1)) return E_ARENA;
        while (b < MAX_FILL && stage_fits(c, &c->st[s], b + 1)) b++;
        c->st[s].bmax = b;
    }
    for (int s = 0; s < c->n_tst; s++) {
        int b = 1;
        if (!tstage_fits(c, &c->tst[s], 1)) return E_ARENA;
        while (b < MAX_FILL && tstage_fits(c, &c->tst[s], b + 1)) b++;
        c->tst[s].bmax = b;
    }
    return E_OK;
}

// compute every stage as far as audio frames [0, F) need
static void advance(itofs_ctx_t *c, int F)
{
    const int S = c->n_stages;
    int need[MAX_STAGES];
    for (;;) {
        for (int s = 0; s < S; s++) need[s] = 0;
        need[S - 1] = imin(F, c->T);
        for (int s = S - 1; s >= 0; s--)
            for (int e = 0; e < c->st[s].np; e++) {
                const int p = c->st[s].prod[e];
                need[p] = imax(need[p], imin(c->T, need[s] + c->st[s].R[e]));
            }
        if (!c->text_incr || c->T_known) break;
        // frames of the text features read by PIN / MIN (reach 0) and the v2 EMB (its own reach): they must be final
        int hf_need = 0;
        for (int s = 0; s < S; s++) if (c->st[s].hfR >= 0) hf_need = imax(hf_need, imin(c->T, need[s] + c->st[s].hfR));
        if (c->start[c->n_done] >= hf_need) break;
        text_ensure(c, hf_need);    // may finish the text side (then T is known and the needs are recomputed)
    }
    for (int s = 0; s < S; s++) {
        itofs_stage_t *st = &c->st[s];
        while (st->done < need[s]) {
            const int t1 = imin(need[s], st->done + st->bmax);
            run_stage(c, s, st->done, t1);
            st->done = t1;
        }
    }
}

// ======================================================================================================================
// API
// ======================================================================================================================
int itofs_begin(itofs_ctx_t *c, const int *tokens, int n, int style_idx, uint32_t seed)
{
    const itofs_model_t *m = c->m;
    if (!tokens || n < 1 || style_idx < 0 || style_idx >= m->n_styles) return E_ARG;
    if (n > c->lim.max_tokens) return E_TOO_LONG;
    for (int i = 0; i < n; i++) if (tokens[i] < 0 || tokens[i] >= m->n_vocab) return E_ARG;
    if (c->act_bits != 8 && c->act_bits != 16) return E_ARG;
    c->act_planes = c->act_bits == 16 ? 2 : 1;
    { const int e = set_bmax(c); if (e) return e; }
    c->n_tok = n;
    memcpy(c->tok, tokens, (size_t)n * sizeof(int));
    if (!m->has_sp) style_setup(c, style_idx);      // with the style predictor: after the text encoder (below)
    // incremental text side: possible when nothing looks at the whole sentence (no backward GRU, no t/T feature)
    c->text_incr = (!m->rnn_hidden || !m->rnn_bidir) && !m->sent_pos && !c->force_whole_text;
    if (!c->text_incr && !c->t_x) return E_ARG;      // whole-sentence path requested but its buffers were not allocated
    if (c->text_incr) {
        c->n_done = 0; c->start[0] = 0;
        c->T = TSENT; c->T_known = 0;
        for (int s = 0; s < c->n_tst; s++) c->tst[s].done = 0;
        if (m->rnn_hidden) memset(c->gru_h, 0, (size_t)m->rnn_hidden * 4);
        if (m->has_sp) { text_h_all(c); sp_style(c, n); }
    } else {
        text_side(c, tokens, n);
        long T = 0;
        for (int i = 0; i < n; i++) {
            const int d = c->force_dur ? imax(1, c->force_dur[i]) : round_dur(itf_expf(c->logd[i]));
            c->dur[i] = d;
            c->start[i] = (int)T;
            T += d;
        }
        if (T >= TSENT) return E_TOO_LONG;
        c->start[n] = (int)T;
        c->T = (int)T; c->T_known = 1; c->n_done = n;
    }
    c->seed = seed;
    c->phase0 = c->use_ext_phase0 ? c->ext_phase0 : (double)mix32(seed ^ 0xa5a5a5a5u) / 4294967296.0;
    c->src_ph = c->phase0;
    c->produced = 0;
    for (int s = 0; s < c->n_stages; s++) c->st[s].done = 0;
    return c->T_known ? c->T : 0;
}

int itofs_next_chunk_f32(itofs_ctx_t *c, float *out, int max_samples)
{
    const int hop = c->m->hop;
    int frames = imin(c->lim.chunk_frames, max_samples / hop);
    if (frames < 1) return E_ARG;
    text_ensure(c, c->produced + frames);              // incremental text side: these frames must exist (or T known)
    if (c->text_incr && !c->T_known && c->start[c->n_done] >= TSENT) return E_TOO_LONG;
    if (c->produced >= c->T) return 0;
    frames = imin(frames, c->T - c->produced);
    c->out_ptr = out;
    c->out_base = c->produced;
    advance(c, c->produced + frames);
    c->produced += frames;
    return frames * hop;
}

int itofs_next_chunk(itofs_ctx_t *c, int16_t *out_pcm, int max_samples)
{
    const int hop = c->m->hop;
    int total = 0;
    while (max_samples - total >= hop) {
        int got = itofs_next_chunk_f32(c, c->s_pcm, imin(max_samples - total, c->lim.chunk_frames * hop));
        if (got <= 0) return total ? total : got;
        for (int i = 0; i < got; i++) {
            float v = c->s_pcm[i] * 32767.f;
            v = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v);
            out_pcm[total + i] = (int16_t)lrintf(v);
        }
        total += got;
        if (c->produced >= c->T) break;
    }
    if (total == 0 && c->produced < c->T) return E_ARG;
    return total;
}
