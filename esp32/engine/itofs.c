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
#define MAX_FACTORS 32
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

// ======================================================================================================================
// deterministic math: only IEEE + - * / (and sqrt / floor, which are exact), so host and chip agree bit for bit
// ======================================================================================================================
#ifdef ITOFS_OPCOUNT
long long itofs_opc[ITOFS_OPC_N];
#define OPC(k, n) (itofs_opc[k] += (n))
#else
#define OPC(k, n) ((void)0)
#endif

static inline float f_from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t bits_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static float itf_expf(float x)
{
    OPC(ITOFS_OPC_EXP, 1);
    if (x != x) return x;
    if (x > 88.7f) return f_from_bits(0x7f800000u);
    if (x < -87.3f) return 0.f;
    const float ln2_hi = 0.693145751953125f, ln2_lo = 1.428606765330187e-06f;
    float kf = floorf(x * 1.44269504088896341f + 0.5f);
    float r = (x - kf * ln2_hi) - kf * ln2_lo;
    float p = 1.f + r * (1.f + r * (0.5f + r * (1.f / 6 + r * (1.f / 24 + r * (1.f / 120 + r * (1.f / 720 + r * (1.f / 5040)))))));
    int k = (int)kf;
    if (k < -125) { p *= f_from_bits(0x00800000u) * 2.f; k += 125; p *= f_from_bits((uint32_t)(k + 127) << 23); return p; }
    return p * f_from_bits((uint32_t)(k + 127) << 23);
}

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

static float itf_erff_h(float x)
{
    OPC(ITOFS_OPC_ERF, 1);
    float a = fabsf(x), r;
    if (a <= 1.f) {
        OPC(ITOFS_OPC_ERF_SERIES, 1);
        const float x2 = a * a;
        float p = ERF_C[12];
        for (int n = 11; n >= 0; n--) p = p * x2 + ERF_C[n];
        r = a * p * 1.12837916709551257f;
    } else if (a < 4.f) {
        float t = 1.f / (1.f + 0.3275911f * a);
        float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        r = 1.f - p * itf_expf(-a * a);
    } else {
        r = 1.f;
    }
    return x < 0 ? -r : r;
}

static void itf_sincosf(float x, float *s, float *c)
{
    OPC(ITOFS_OPC_SINCOS, 1);
    const float p1 = 1.5703125f, p2 = 4.837512969970703125e-4f, p3 = 7.54978995489188216e-8f;
    float kf = floorf(x * 0.636619772367581343f + 0.5f);
    float r = ((x - kf * p1) - kf * p2) - kf * p3;
    float r2 = r * r;
    float sn = r * (1.f + r2 * (-1.f / 6 + r2 * (1.f / 120 + r2 * (-1.f / 5040 + r2 * (1.f / 362880 + r2 * (-1.f / 39916800))))));
    float cs = 1.f + r2 * (-0.5f + r2 * (1.f / 24 + r2 * (-1.f / 720 + r2 * (1.f / 40320 + r2 * (-1.f / 3628800 + r2 * (1.f / 479001600))))));
    switch (((int)kf) & 3) {
    case 0: *s = sn; *c = cs; break;
    case 1: *s = cs; *c = -sn; break;
    case 2: *s = -sn; *c = -cs; break;
    default: *s = -cs; *c = sn; break;
    }
}

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
    float *ring;
    int done;                      // frames [0, done) computed
};

enum { TK_ENC, TK_GRU, TK_PROJ, TK_DUR, TK_DOUT };
#define MAX_TSTAGES (2 * ITOFS_MAX_LAYERS + 4)
struct itofs_tstage { int kind, idx, L, R, width, done; float *ring; };

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

static int max_qin, max_qout_tile, max_k;
static void scan_q(const itofs_qlin_t *q)
{
    if (!q->w) return;
    max_qin = imax(max_qin, q->in);
    max_qout_tile = imax(max_qout_tile, imin(q->out, ACC_TILE));
    max_k = imax(max_k, q->K);
}

static void layout(itofs_ctx_t *c, const itofs_model_t *m, const itofs_limits_t *lim, bump_t *hot, bump_t *bulk)
{
    const int D = m->text_dim, H = m->rnn_hidden, PD = m->pros_dim, DD = m->dec_dim, N = m->n_fft, M = N / 2;
    const int Lmax = lim->max_tokens, nb = lim->chunk_frames;
    max_qin = max_qout_tile = max_k = 0;
    for (int l = 0; l < m->text_layers; l++) scan_q(&m->enc[l].conv);
    for (int l = 0; l < m->dur_layers; l++) scan_q(&m->dur[l].conv);
    for (int l = 0; l < m->pros_layers; l++) scan_q(&m->pros[l].conv);
    if (H) { scan_q(&m->rnn_ih[0]); scan_q(&m->rnn_ih[1]); scan_q(&m->rnn_proj); }
    scan_q(&m->pros_in); scan_q(&m->dec_embed); scan_q(&m->harm_proj); scan_q(&m->dec_out);
    if (m->arch == 3) { scan_q(&m->mel_in); scan_q(&m->mel_out); for (int l = 0; l < m->mel_layers; l++) { scan_q(&m->mel[l].conv); scan_q(&m->mel[l].film_q); } }
    for (int i = 0; i < 3 && m->has_sp; i++) scan_q(&m->sp[i]);
    scan_q(&m->style_q);
    for (int l = 0; l < m->dur_layers; l++) scan_q(&m->dur[l].film_q);
    for (int l = 0; l < m->pros_layers; l++) scan_q(&m->pros[l].film_q);
    const int MD = m->arch == 3 ? m->mel_dim : 0, MIN_IN = m->arch == 3 ? m->hf_dim + 3 : 0, NM = m->arch == 3 ? m->n_mels : 0;
    for (int b = 0; b < m->dec_blocks; b++) { scan_q(&m->blk[b].pw1); scan_q(&m->blk[b].pw2); }
    max_k = imax(max_k, m->dec_kernel);             // the depthwise convs gather K rows too
    const int grow = nb + max_k - 1;
    const int whole_txt = (m->rnn_hidden && m->rnn_bidir) || m->sent_pos || lim->whole_text;
    const int tb = whole_txt ? imax(nb, TEXT_BATCH) : nb;                       // whole-sentence text-side batch
    const int tk = imax(imax(m->text_kernel, m->dur_kernel), 1), growt = tb + tk - 1, tin = imax(D, 2 * H);
    const int gw = imax(imax(imax(imax(PD, m->dec_in), imax(DD, m->hf_dim)), imax(D, H)), imax(MD, MIN_IN));
    // y0: prosody / embed / block rows, the harmonic STFT rows and the head output (nb x (n_fft+2));
    // y1: one STFT frame, the ConvNeXt hidden (nb x inter) and one half spectrum; y2: harm_proj / pw2 outputs
    const int y0w = imax(imax(imax(imax(N + 2, DD), PD), imax(D, 3 * H)), imax(MD, NM)), y1n = imax(nb * m->dec_inter, N + 2), y2w = imax(DD, PD);
    // ---- hot scratch
    c->s_g0 = bump(hot, (size_t)grow * gw * 4);
    c->s_g1 = bump(hot, (size_t)grow * gw * 4);
    c->s_y0 = bump(hot, (size_t)nb * y0w * 4);
    c->s_y1 = bump(hot, (size_t)y1n * 4);
    c->s_y2 = bump(hot, (size_t)nb * y2w * 4);
    c->s_qs = bump(hot, (size_t)imax(grow, growt) * 4);
    c->s_q8 = bump(hot, (size_t)imax(grow * PLANES * pad16(max_qin), growt * PLANES * pad16(tin)) + 64);
    c->acc_bytes = (size_t)nb * PLANES * max_qout_tile * 2 * 4;
    c->s_acc = bump(hot, c->acc_bytes);   // x 2 weight planes (int16 weights)
    c->tb = tb;
    c->fbuf = bump(hot, (size_t)(N + 8) * 4);
    c->win = bump(hot, (size_t)N * 4);
    c->fft = bump(hot, sizeof(itofs_fft_t));
    cpx *tmp = bump(hot, sizeof(cpx) * (size_t)M);
    // read-only tables (read sequentially, cache friendly): bulk, to keep the hot arena in internal SRAM
    c->win2 = bump(bulk, (size_t)N * 4);
    c->rtw = bump(bulk, (size_t)(M + 1) * 2 * 4);
    cpx *tw = bump(bulk, sizeof(cpx) * (size_t)M);
    if (c->fft) { c->fft->tw = tw; c->fft->tmp = tmp; }
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
    c->fft->n = M;
    if (fft_factor(M, c->fft->fac)) return E_CONFIG;
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
// q = 128*hi + lo); P = 1: one int8 plane, q = floor(x * 127 / max|x| + 0.5). Returns the scale.
// floor(v) for |v| < 2^31 without a libm call (identical result)
static inline float ffloor_small(float v) { int i = (int)v; float f = (float)i; return f > v ? f - 1.f : f; }

static float quant_row(const float *x, int in, int ldq, int P, int8_t *q)
{
    OPC(ITOFS_OPC_QUANT, in);
    float mx = 0.f;
    for (int i = 0; i < in; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    if (!(mx > 0.f)) { memset(q, 0, (size_t)ldq * P); return 0.f; }
    if (P == 1) {
        const float Q = QMAX8, inv = Q / mx;
        for (int i = 0; i < in; i++) {
            float v = ffloor_small(x[i] * inv + 0.5f);   // |x * inv| <= 127 (+ rounding)
            v = v > Q ? Q : (v < -Q ? -Q : v);
            q[i] = (int8_t)(int)v;
        }
        return mx / Q;
    }
    const float Q = QMAX16, inv = Q / mx;
    int8_t *hi = q, *lo = q + ldq;
    for (int i = 0; i < in; i++) {
        float v = ffloor_small(x[i] * inv + 0.5f);       // |x * inv| <= 16256 (+ rounding)
        v = v > Q ? Q : (v < -Q ? -Q : v);
        int qi = (int)v;
        int hh = (qi + 64 + 16384) / 128 - 128;       // floor((qi + 64) / 128)
        hi[i] = (int8_t)hh;
        lo[i] = (int8_t)(qi - 128 * hh);
    }
    return mx / Q;
}

// Dense int8 / int16-weight conv or linear on n rows. X holds n + K - 1 rows (already zero padded), row stride ldx;
// the first L->in values of each row are the input. Y: n rows, stride ldy.
// Activations: P planes (int16-weight layers always 2, int8-weight layers c->act_planes); weights: WP planes.
// Every (activation plane, weight plane) pair is one exact int8 GEMM; they are combined exactly in integers.
static void qlin_run(itofs_ctx_t *c, const itofs_qlin_t *L, const float *X, int ldx, int n, float *Y, int ldy, int grp)
{
    const int in = L->in, out = L->out, K = L->K, ldq = pad16(in);
    const int WP = L->w_lo ? 2 : 1, P = (L->w_lo || L->act16) ? 2 : c->act_planes;
    const int rows = n + K - 1;
    for (int r = 0; r < rows; r++)
        c->s_qs[r] = quant_row(X + (size_t)r * ldx, in, ldq, P, c->s_q8 + (size_t)r * P * ldq);
    for (int t = 0; t < n; t++) memset(Y + (size_t)t * ldy, 0, (size_t)out * 4);
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
            c->qgemm(xq, n * P, ldq, in, L->w + woff, ot, a0, c->qgemm_user);
            if (WP == 2) c->qgemm(xq, n * P, ldq, in, L->w_lo + woff, ot, a1, c->qgemm_user);
            for (int t = 0; t < n; t++) {
                const float s = c->s_qs[t + k];
                if (s == 0.f) continue;
                float *y = Y + (size_t)t * ldy + o0;
                if (WP == 1 && P == 1) {
                    const int32_t *a = a0 + (size_t)t * ot;
                    for (int o = 0; o < ot; o++) y[o] += s * (float)a[o];
                } else if (WP == 1) {
                    const int32_t *ah = a0 + (size_t)(2 * t) * ot, *al = ah + ot;
                    if (exact32) for (int o = 0; o < ot; o++) y[o] += s * (float)(128 * ah[o] + al[o]);
                    else for (int o = 0; o < ot; o++) y[o] += s * (float)((int64_t)ah[o] * 128 + al[o]);
                } else {                // int16 weights x 15-bit activations: 4 exact products
                    const int32_t *hh = a0 + (size_t)(2 * t) * ot, *lh = hh + ot;     // (x hi, x lo) x w hi
                    const int32_t *hl = a1 + (size_t)(2 * t) * ot, *ll = hl + ot;     // (x hi, x lo) x w lo
                    for (int o = 0; o < ot; o++) {
                        const int64_t v = ((int64_t)hh[o] * 128 + lh[o]) * 256 + ((int64_t)hl[o] * 128 + ll[o]);
                        y[o] += s * (float)v;
                    }
                }
            }
        }
    }
    for (int t = 0; t < n; t++) {
        float *y = Y + (size_t)t * ldy;
        if (L->b) for (int o = 0; o < out; o++) y[o] = y[o] * L->sw[o] + L->b[o];
        else for (int o = 0; o < out; o++) y[o] = y[o] * L->sw[o];
    }
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
static inline float gelu_v3(float x) { return 0.5f * x * (1.0f + itf_erff_h(x * 0.70710678118654752f)); }
#define gelu(x) (m_arch3 ? gelu_v3(x) : gelu_v2(x))

// y[o] = b[o] + sum_i W[o][i] x[i]  (f32)
static void gemv_f32(itofs_ctx_t *c, const float *W, const float *b, const float *x, int out, int in, float *y, int grp)
{
    for (int o = 0; o < out; o++) {
        const float *w = W + (size_t)o * in;
        float a = 0.f;
        for (int i = 0; i < in; i++) a += w[i] * x[i];
        y[o] = a + (b ? b[o] : 0.f);
    }
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

static void run_tstage(itofs_ctx_t *c, int s, int i0, int i1)
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
        for (int t = 0; t < n; t++) {
            float *y = c->s_y0 + (size_t)t * D, *o = tring_row(c, s, i0 + t);
            const float *xr = c->s_g0 + (size_t)(t + S->L) * D;
            layernorm_row(y, D, L->ln_g, L->ln_b, m->eps_text, y);
            if (g1) for (int i = 0; i < D; i++) y[i] = y[i] * g1[i] + bb[i];
            for (int i = 0; i < D; i++) o[i] = xr[i] + gelu(y[i]);
        }
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
            const int i1 = imin(need[s], st->done + c->nb);
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
    const int step = c->text_step > 0 ? c->text_step : c->nb;
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

static float gauss_at(uint32_t seed, int n)
{
    OPC(ITOFS_OPC_GAUSS, 1);
    const uint32_t pair = (uint32_t)n >> 1;
    const uint32_t a = mix32(seed ^ mix32(2u * pair + 0x9e3779b9u)), b = mix32(seed ^ mix32(2u * pair + 1u + 0x9e3779b9u) ^ 0x85ebca6bu);
    const float u1 = ((float)(a >> 8) + 0.5f) * (1.0f / 16777216.0f), u2 = ((float)(b >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float r = sqrtf(-2.f * itf_logf(u1));
    float s, co;
    itf_sincosf(6.28318530717958648f * u2, &s, &co);
    return (n & 1) ? r * s : r * co;
}

// ======================================================================================================================
// stage runners
// ======================================================================================================================
static inline float *ring_row(const itofs_ctx_t *c, int s, int t)
{
    return c->st[s].ring + (size_t)(t & (c->st[s].rframes - 1)) * c->st[s].width;
}

// rows [t0, t1) of producer p (w floats from column off), zero outside [0, T)
static void gather(itofs_ctx_t *c, int p, int t0, int t1, int off, int w, float *dst, int ldd)
{
    for (int t = t0; t < t1; t++) {
        float *d = dst + (size_t)(t - t0) * ldd;
        if (t < 0 || t >= c->T) memset(d, 0, (size_t)w * 4);
        else memcpy(d, ring_row(c, p, t) + off, (size_t)w * 4);
    }
}

static void put_rows(itofs_ctx_t *c, int s, int t0, int n, const float *src, int lds)
{
    const int w = c->st[s].width;
    for (int t = 0; t < n; t++) memcpy(ring_row(c, s, t0 + t), src + (size_t)t * lds, (size_t)w * 4);
}

// one piece of the source: samples n0 .. n0+len-1 whose F0 is interpolated between a (at w=0) and b (at w=1) with
// w_j = (j + 0.5) / hop, j = j0 .. j0+len-1 (constant: f = a). Phase anchored in double at the piece start.
static void src_piece(itofs_ctx_t *c, int n0, int len, float a, float b, int j0, int constant, float *out)
{
    OPC(ITOFS_OPC_SRC, len);
    const itofs_model_t *m = c->m;
    const float hop = (float)m->hop, sr = (float)m->sr, half_sr = 0.5f * (float)m->sr;
    const float d = constant ? 0.f : (b - a) / hop;
    const double P = c->src_ph;
    const float Pf = (float)(P - floor(P));
    const float f_first = constant ? a : a + d * ((float)j0 + 0.5f);
    const float amp_uv = m->src_amp / 3.0f;
    for (int J = 0; J < len; J++) {
        const float j = (float)(j0 + J);
        float fl, f;
        if (constant) { fl = a; f = a; }
        else {
            const float w = (j + 0.5f) / hop;
            fl = a + d * (j + 0.5f);                 // same line, closed form for the phase sum
            f = a * (1.f - w) + b * w;               // torch's form for the voicing / Nyquist tests
        }
        // phase after this sample: P + sum_{m<=J} f_m / sr (arithmetic series)
        float phi = Pf + ((float)(J + 1) * (f_first + fl) * 0.5f) / sr;
        phi = phi - floorf(phi);
        float s1, c1;
        itf_sincosf(6.28318530717958648f * phi, &s1, &c1);
        float sk = s1, ck = c1, hsum = 0.f;
        for (int k = 1; k <= m->n_harm; k++) {
            if ((float)k * f < half_sr) hsum += sk;
            const float ns = sk * c1 + ck * s1, nc = ck * c1 - sk * s1;
            sk = ns; ck = nc;
        }
        const int n = n0 + J;
        const float nz = c->ext_noise ? c->ext_noise[n] : gauss_at(c->seed, n);
        const int uv = f > m->uv_hz;
        out[J] = uv ? (m->src_amp * (hsum / (float)m->n_harm) + m->src_noise * nz) : amp_uv * nz;
    }
    // advance the double anchor by the exact sum of the piece
    const double da = a, db = b, dd = constant ? 0.0 : (db - da) / (double)m->hop;
    const double f0d = constant ? da : da + dd * ((double)j0 + 0.5), f1d = constant ? da : da + dd * ((double)(j0 + len - 1) + 0.5);
    c->src_ph = P + (double)len * (f0d + f1d) * 0.5 / (double)m->sr;
}

static void src_frame(itofs_ctx_t *c, int s, int t, float *out)
{
    const int hop = c->m->hop, T = c->T, h2 = hop / 2;
    const int pc = c->st[s].prod[0];
    // piece A: samples [t*hop, t*hop + hop/2) -> segment t-1 (constant f0[0] before the first frame centre)
    if (t == 0) src_piece(c, 0, h2, ring_row(c, pc, 0)[3], 0.f, 0, 1, out);
    else src_piece(c, t * hop, h2, ring_row(c, pc, t - 1)[3], ring_row(c, pc, t)[3], h2, 0, out);
    // piece B: samples [t*hop + hop/2, (t+1)*hop) -> segment t (constant f0[T-1] after the last frame centre)
    if (t == T - 1) src_piece(c, t * hop + h2, hop - h2, ring_row(c, pc, t)[3], 0.f, 0, 1, out + h2);
    else src_piece(c, t * hop + h2, hop - h2, ring_row(c, pc, t)[3], ring_row(c, pc, t + 1)[3], 0, 0, out + h2);
}

// forward real FFT of N windowed samples -> row [re(0..M), im(0..M)]
static void rfft_row(itofs_ctx_t *c, const float *x, float *row)
{
    const int N = c->m->n_fft, M = N / 2;
    cpx *z = (cpx *)c->fbuf;
    for (int n = 0; n < M; n++) { z[n].r = x[2 * n]; z[n].i = x[2 * n + 1]; }
    fft_fwd(c->fft, z);
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
static void irfft_win(itofs_ctx_t *c, const float *X, float *out)
{
    const int N = c->m->n_fft, M = N / 2;
    cpx *buf = (cpx *)c->fbuf;
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
    fft_fwd(c->fft, buf);
    const float invM = 1.0f / (float)M;
    for (int j = 0; j < M; j++) {
        out[2 * j] = buf[j].r * invM * c->win[2 * j];
        out[2 * j + 1] = -buf[j].i * invM * c->win[2 * j + 1];
    }
}

static void run_stage(itofs_ctx_t *c, int s, int t0, int t1)
{
    const itofs_model_t *m = c->m;
    const int m_arch3 = m->arch == 3;
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
        for (int t = 0; t < n; t++) {
            float *y = c->s_y0 + (size_t)t * W;
            const float *xr = c->s_g0 + (size_t)(t + lp) * W;
            layernorm_row(y, W, L->ln_g, L->ln_b, m->eps_text, y);
            for (int i = 0; i < W; i++) y[i] = xr[i] + gelu(y[i] * g1[i] + bb[i]);
        }
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
        for (int t = t0; t < t1; t++) {
            float *r = ring_row(c, s, t);
            src_frame(c, s, t, r);
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_SRC, t, r, W);
        }
        break;
    }
    case SK_HFT: {
        const int hop = m->hop, N = m->n_fft, T = c->T, p = S->prod[0];
        for (int t = t0; t < t1; t++) {
            float *x = c->s_y1;             // N windowed samples
            for (int j = 0; j < N; j++) {
                int zi = t * hop + j - N / 2;
                if (zi < 0) zi = -zi;       // reflect pad (center=True)
                const int si = zi - hop / 2; // hop/2 zeros each side of the source
                const float v = (si < 0 || si >= T * hop) ? 0.f : ring_row(c, p, si / hop)[si % hop];
                x[j] = v * c->win[j];
            }
            float *row = c->s_y0 + (size_t)(t - t0) * (N + 2);
            rfft_row(c, x, row);
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_HFEAT, t, row, N + 2);
        }
        qlin_run(c, &m->harm_proj, c->s_y0, N + 2, n, c->s_y2, W, ITOFS_G_HARM);
        put_rows(c, s, t0, n, c->s_y2, W);
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
        for (int t = 0; t < n; t++) {
            float *y = c->s_y0 + (size_t)t * W;
            if (m->arch == 2) for (int i = 0; i < W; i++) y[i] += c->cond[i];
            layernorm_row(y, W, m->n0_g, m->n0_b, m->eps_dec, y);
            const float *hp = ring_row(c, ph, t0 + t);
            for (int i = 0; i < W; i++) y[i] += hp[i];
        }
        put_rows(c, s, t0, n, c->s_y0, W);
        break;
    }
    case SK_BLK: {
        const itofs_block_t *bk = &m->blk[S->idx];
        const int K = m->dec_kernel, lp = S->L[0], DI = m->dec_inter;
        gather(c, S->prod[0], t0 - lp, t1 + S->R[0], 0, W, c->s_g0, W);
        for (int t = 0; t < n; t++) {
            float *y = c->s_y0 + (size_t)t * W;
            for (int ch = 0; ch < W; ch++) {
                float a = 0.f;
                const float *w = bk->dw_w + (size_t)ch * K;
                for (int k = 0; k < K; k++) a += c->s_g0[(size_t)(t + k) * W + ch] * w[k];
                y[ch] = a + bk->dw_b[ch];
            }
            layernorm_row(y, W, bk->ln_g, bk->ln_b, m->eps_dec, y);
        }
        c->macs_f32[ITOFS_G_BLOCKS] += (double)n * W * K;
        qlin_run(c, &bk->pw1, c->s_y0, W, n, c->s_y1, DI, ITOFS_G_BLOCKS);
        for (size_t j = 0; j < (size_t)n * DI; j++) c->s_y1[j] = gelu(c->s_y1[j]);
        qlin_run(c, &bk->pw2, c->s_y1, DI, n, c->s_y2, W, ITOFS_G_BLOCKS);
        for (int t = 0; t < n; t++) {
            float *y = c->s_y2 + (size_t)t * W;
            const float *r = c->s_g0 + (size_t)(t + lp) * W;
            for (int ch = 0; ch < W; ch++) y[ch] = r[ch] + y[ch] * bk->gamma[ch];
        }
        put_rows(c, s, t0, n, c->s_y2, W);
        break;
    }
    case SK_HEAD: {
        const int D = m->dec_dim, N = m->n_fft, NB = N / 2 + 1;
        gather(c, S->prod[0], t0, t1, 0, D, c->s_g0, D);
        for (int t = 0; t < n; t++) layernorm_row(c->s_g0 + (size_t)t * D, D, m->n1_g, m->n1_b, m->eps_dec, c->s_g0 + (size_t)t * D);
        qlin_run(c, &m->dec_out, c->s_g0, D, n, c->s_y0, N + 2, ITOFS_G_HEAD);
        for (int t = 0; t < n; t++) {
            const float *h = c->s_y0 + (size_t)t * (N + 2);
            if (c->tap) c->tap(c->tap_user, ITOFS_TAP_SPEC, t0 + t, h, N + 2);
            float *X = c->s_y1;              // interleaved half spectrum
            for (int k = 0; k < NB; k++) {
                float mag = itf_expf(h[k]);
                if (mag > m->mag_max) mag = m->mag_max;
                float sn, cs;
                itf_sincosf(h[NB + k], &sn, &cs);
                X[2 * k] = mag * cs;
                X[2 * k + 1] = mag * sn;
            }
            X[1] = 0.f; X[2 * (NB - 1) + 1] = 0.f;    // c2r ignores imag(DC), imag(Nyquist)
            irfft_win(c, X, ring_row(c, s, t0 + t));
        }
        break;
    }
    case SK_OLA: {
        const int hop = m->hop, N = m->n_fft, p = S->prod[0];
        for (int t = t0; t < t1; t++) {
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
        break;
    }
    }
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
            const int t1 = imin(need[s], st->done + c->nb);
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
