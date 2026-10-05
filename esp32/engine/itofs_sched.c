// See itofs_sched.h. Pure C99: no timing source, no allocation.
#include "itofs_sched.h"
#include <stddef.h>
#include <math.h>

static void sort_d(double *a, int n)         // insertion sort: n <= ITOFS_SCHED_MAXCH
{
    for (int i = 1; i < n; i++) { const double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
}
static double median_d(double *a, int n) { sort_d(a, n); return n & 1 ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]); }
static double audio_us(const itofs_cal_t *c, int frames) { return (double)frames * c->hop / c->sr * 1e6; }

double itofs_cal_rtf(const itofs_cal_t *c, double *steady_us, double *whole)
{
    double r[ITOFS_SCHED_MAXCH], ts[ITOFS_SCHED_MAXCH], sum_t = 0, sum_a = 0;
    int nf = 0;
    for (int k = 0; k < c->n && k < ITOFS_SCHED_MAXCH; k++) {
        const double a = audio_us(c, c->frames[k]);
        sum_t += c->t_us[k]; sum_a += a;
        if (c->frames[k] == c->steady_frames) { r[nf] = c->t_us[k] / a; ts[nf] = c->t_us[k]; nf++; }
    }
    if (whole) *whole = sum_a > 0 ? sum_t / sum_a : -1;
    if (nf == 0) { if (steady_us) *steady_us = 0; return -1; }
    const double m = median_d(r, nf), w = sum_a > 0 ? sum_t / sum_a : 0;
    if (steady_us) *steady_us = median_d(ts, nf);
    return m > w ? m : w;
}

int itofs_sched_pick(const double *rtf, int n, double target, double degraded, itofs_tier_status_t *status)
{
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (rtf[i] < 0) continue;
        if (rtf[i] <= target) { if (status) *status = ITOFS_TIER_OK; return i; }
        if (best < 0 || rtf[i] < rtf[best]) best = i;
    }
    if (best >= 0 && status) *status = rtf[best] >= degraded ? ITOFS_TIER_DEGRADED : ITOFS_TIER_MARGINAL;
    return best;
}

int itofs_sched_simulate(const double *t_us, const double *a_us, int n, double begin_us, double start_delay_us, int nbuf, double *end_us)
{
    double ring[16];                                                     // playback end of the last nbuf chunks
    int under = 0;
    double prod = begin_us;                                              // when synthesis is free to start chunk k
    double pe_prev = 0;
    if (nbuf > 16) nbuf = 16;
    for (int k = 0; k < n; k++) {
        double start = prod;
        if (nbuf > 0 && k >= nbuf && ring[k % nbuf] > start) start = ring[k % nbuf];     // no free PCM buffer until chunk k - nbuf has played
        const double done = start + t_us[k];
        prod = done;
        double ps;
        if (k == 0) ps = done > start_delay_us ? done : start_delay_us;
        else {
            ps = pe_prev;
            if (done > ps + 1e-6) { under++; ps = done; }               // the speaker plays silence until the chunk arrives
        }
        pe_prev = ps + a_us[k];
        if (nbuf > 0) ring[k % nbuf] = pe_prev;
    }
    if (end_us) *end_us = n ? pe_prev : 0;
    return under;
}

// chunk list of an utterance: frames, audio and (model) production time
static int build(const itofs_cal_t *cal, int total_frames, double safety, const double *obs, int n_obs, double *fr, double *t, double *a)
{
    double ts[ITOFS_SCHED_MAXCH];
    int nf = 0;
    for (int k = 0; k < cal->n; k++) if (cal->frames[k] == cal->steady_frames) ts[nf++] = cal->t_us[k];
    const double t_steady = nf ? median_d(ts, nf) : 0.0;
    // time model t = fa + fb * frames from the whole trace (least squares, clamped), for chunks off the trace and for the short last chunk
    double sx = 0, sy = 0, sxx = 0, sxy = 0; const int m = cal->n;
    for (int k = 0; k < m; k++) { sx += cal->frames[k]; sy += cal->t_us[k]; sxx += (double)cal->frames[k] * cal->frames[k]; sxy += cal->frames[k] * cal->t_us[k]; }
    double fb = 0, fa = nf ? t_steady : 0;
    if (m >= 2 && m * sxx - sx * sx > 1e-9) {
        fb = (m * sxy - sx * sy) / (m * sxx - sx * sx); fa = (sy - fb * sx) / m;
        if (fb < 0) { fb = 0; fa = sy / m; }
        if (fa < 0) { fa = 0; fb = sy / sx; }
    }
    int n = 0, left = total_frames;
    while (left > 0 && n < ITOFS_SCHED_PLANCH) {
        int f = n < cal->n ? cal->frames[n] : cal->steady_frames;
        if (f > left) f = left;
        fr[n] = f; a[n] = audio_us(cal, f);
        double tk;
        if (n < n_obs && obs) tk = obs[n];
        else if (n < cal->n && cal->frames[n] == f) tk = cal->t_us[n] * safety;
        else if (f == cal->steady_frames && nf) tk = t_steady * safety;
        else tk = (fa + fb * f) * safety;
        t[n] = tk; n++; left -= f;
    }
    return n;
}

void itofs_sched_plan(const itofs_cal_t *cal, int n_tok, int n_frames_hint, int nbuf, double safety, double max_delay_us,
                      const double *obs_t_us, int n_obs, itofs_plan_t *out)
{
    double fr[ITOFS_SCHED_PLANCH], t[ITOFS_SCHED_PLANCH], a[ITOFS_SCHED_PLANCH];     // (on the stack: 3 x 48 doubles)
    double steady = 0, whole = 0;
    out->rtf = itofs_cal_rtf(cal, &steady, &whole);
    int tot = 0;
    for (int k = 0; k < cal->n; k++) tot += cal->frames[k];
    const double fpt = cal->n_tok > 0 ? (double)tot / cal->n_tok : 2.7;
    int total = n_frames_hint > 0 ? n_frames_hint : (int)(n_tok * fpt + 0.5);
    if (total < 1) total = 1;
    double sf = safety;                                    // a margin cannot be allowed to turn a real-time set into a "slower than real time" one in the model
    {
        const double rs = steady > 0 ? steady / audio_us(cal, cal->steady_frames) : 0;
        if (rs > 0 && rs * sf > 0.985) sf = 0.985 / rs > 1.0 ? 0.985 / rs : 1.0;
    }
    const int n = build(cal, total, sf, obs_t_us, n_obs, fr, t, a);
    out->nchunks = n;
    const double begin = cal->begin_us * sf;
    out->first_audio_us = begin + t[0];
    // The delay with the fewest predicted underruns, the smallest such delay. Underruns are NOT monotone in the delay: a longer wait lets the
    // synthesis fill every PCM buffer and then stall until playback starts, so the chunks after them can arrive late. A 2 ms grid (10 ms past 400 ms) is searched; the usual case stops at the first zero within a few steps.
    double s_best = out->first_audio_us;
    int u_min = 1 << 30;
    for (double d = out->first_audio_us; d <= max_delay_us + 1.0; d += (d - out->first_audio_us < 400000.0 ? 2000.0 : 10000.0)) {   // 2 ms steps for the first 400 ms, then 10 ms
        const int u = itofs_sched_simulate(t, a, n, begin, d, nbuf, NULL);
        if (u < u_min) { u_min = u; s_best = d; if (u == 0) break; }
    }
    out->start_delay_us = s_best;
    out->underruns = u_min;
    out->feasible = u_min == 0;
}
