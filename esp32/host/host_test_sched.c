// Host test of engine/itofs_sched.c with SIMULATED timings (no board needed): the weight-set choice, the boot RTF measurement and the playback
// start delay, against a discrete-event model of the speaker that is independent of the planner's own bookkeeping (it plays the TRUE chunk times of
// a different sentence, with its own jitter, through the same buffer pool).
//   make test_sched
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itofs_sched.h"

#define HOP 300
#define SR 24000
#define NBUF 6
static uint32_t rs = 12345u;
static double urand(void) { rs = rs * 1664525u + 1013904223u; return (rs >> 8) / 16777216.0; }

// A synthetic board: chunk time = (A + B * frames) * (1 + text-step extra in the first chunks) * sentence factor * (1 + jitter)
typedef struct { double rho, alpha, jitter, sent_sd; } world_t;     // steady chunk time = rho * 300 ms; fixed part alpha of it
static const int RAMP[] = {10, 11, 12, 14, 18};
static int frames_of(int k) { return k < 5 ? RAMP[k] : 24; }
static double true_time(const world_t *w, int k, int frames, double sent_factor)
{
    const double t24 = w->rho * 24 * HOP / (double)SR * 1e6;
    const double A = w->alpha * t24, B = (t24 - A) / 24.0;
    double t = A + B * frames;
    if (k < 8) t *= 1.08;                                 // chunks that carry a text-side step
    return t * sent_factor * (1.0 + w->jitter * (2 * urand() - 1));
}
static void make_utt(const world_t *w, int total_frames, double *t, double *a, int *fr, int *n, double sent_factor, double *begin)
{
    int k = 0, left = total_frames;
    while (left > 0) { int f = frames_of(k); if (f > left) f = left; fr[k] = f; a[k] = f * (double)HOP / SR * 1e6; t[k] = true_time(w, k, f, sent_factor); left -= f; k++; }
    *n = k;
    *begin = 6000.0 * sent_factor;
}
static void calibrate(const world_t *w, int total_frames, itofs_cal_t *c, int ntok)
{
    double t[256], a[256], b; int fr[256], n;
    make_utt(w, total_frames, t, a, fr, &n, 1.0 + w->sent_sd * (2 * urand() - 1), &b);
    memset(c, 0, sizeof *c);
    c->n = n; c->hop = HOP; c->sr = SR; c->steady_frames = 24; c->n_tok = ntok; c->begin_us = b;
    for (int k = 0; k < n; k++) { c->frames[k] = fr[k]; c->t_us[k] = t[k]; }
}

int main(void)
{
    int fails = 0;
    // ---- 1. weight-set choice
    {
        struct { double r[4]; int n, want; itofs_tier_status_t st; const char *what; } T[] = {
            {{0.70, 0.60, 0.50}, 3, 0, ITOFS_TIER_OK, "main already fine"},
            {{0.90, 0.80, 0.70}, 3, 1, ITOFS_TIER_OK, "second set fits"},
            {{0.95, 0.90, 0.80}, 3, 2, ITOFS_TIER_OK, "light set fits"},
            {{0.95, 0.90, 0.86}, 3, 2, ITOFS_TIER_MARGINAL, "none fits, best is marginal"},
            {{1.20, 1.05, 0.97}, 3, 2, ITOFS_TIER_DEGRADED, "none fits, best >= 0.95: degraded"},
            {{1.20, 1.05, 0.95}, 3, 2, ITOFS_TIER_DEGRADED, "exactly 0.95 is degraded"},
            {{0.85, 0.80, 0.70}, 3, 0, ITOFS_TIER_OK, "exactly 0.85 is accepted"},
            {{0.86, -1.0, 0.84}, 3, 2, ITOFS_TIER_OK, "a missing set is skipped"},
            {{-1.0, -1.0, -1.0}, 3, -1, ITOFS_TIER_OK, "nothing measured"},
            {{0.94, 0.93, 0.92}, 3, 2, ITOFS_TIER_MARGINAL, "lowest of three marginal sets"},
        };
        for (unsigned i = 0; i < sizeof T / sizeof T[0]; i++) {
            itofs_tier_status_t st = ITOFS_TIER_OK;
            const int got = itofs_sched_pick(T[i].r, T[i].n, ITOFS_RTF_TARGET, ITOFS_RTF_DEGRADED, &st);
            const int ok = got == T[i].want && (got < 0 || st == T[i].st);
            printf("pick %-34s -> set %d status %d %s\n", T[i].what, got, (int)st, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    }
    // ---- 2. measured RTF of a trace
    {
        world_t w = {0.60, 0.5, 0.0, 0.0};
        itofs_cal_t c; calibrate(&w, 450, &c, 84);
        double st, wh;
        const double r = itofs_cal_rtf(&c, &st, &wh);
        const int ok = r >= 0.60 - 1e-9 && r < 0.75 && wh >= 0.60;
        printf("RTF of a trace built at rho 0.60: steady/whole max %.3f (whole %.3f) %s\n", r, wh, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    // ---- 3. no underrun for accepted sets: many random boards, sentences, jitter
    const double rhos[] = {0.30, 0.45, 0.55, 0.65, 0.75, 0.85};
    const double alphas[] = {0.2, 0.35, 0.5};
    const int lens[] = {3, 8, 20, 40, 84, 150, 250, 400};
    long trials = 0, bad = 0; double worst_delay = 0, sum_extra = 0;
    for (unsigned ri = 0; ri < sizeof rhos / sizeof rhos[0]; ri++)
        for (unsigned ai = 0; ai < 3; ai++)
            for (int rep = 0; rep < 60; rep++) {
                world_t w = {rhos[ri], alphas[ai], 0.04, 0.04};     // chunk jitter +-4 %, sentence-to-sentence cost +-4 %
                itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
                for (unsigned li = 0; li < sizeof lens / sizeof lens[0]; li++) {
                    const int total = (int)(lens[li] * 450 / 84.0 + 0.5);
                    double t[256], a[256], b; int fr[256], n;
                    make_utt(&w, total, t, a, fr, &n, 1.0 + w.sent_sd * (2 * urand() - 1), &b);
                    itofs_plan_t p;
                    itofs_sched_plan(&cal, lens[li], 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, NULL, 0, &p);
                    double end;
                    const int u = itofs_sched_simulate(t, a, n, b, p.start_delay_us, NBUF, &end);
                    trials++;
                    if (u || !p.feasible) { bad++; if (bad <= 5) printf("  UNDERRUN rho %.2f alpha %.1f tokens %d: %d underruns, delay %.0f ms, feasible %d\n", w.rho, w.alpha, lens[li], u, p.start_delay_us / 1e3, p.feasible); }
                    if (p.start_delay_us > worst_delay) worst_delay = p.start_delay_us;
                    sum_extra += p.start_delay_us - p.first_audio_us;
                }
            }
    printf("accepted sets (rho <= 0.85), jitter +-4%%, sentence cost +-4%%: %ld sentences played, %ld with an underrun; longest start delay %.0f ms; mean wait beyond the first chunk %.0f ms -> %s\n",
           trials, bad, worst_delay / 1e3, sum_extra / trials / 1e3, bad ? "FAIL" : "ok");
    fails += bad != 0;
    // ---- 3b. beyond the margin (information): jitter +-8 %, sentence cost +-8 % (calibration and playback differ by up to 1.37x)
    {
        long n = 0, nb = 0;
        for (unsigned ri = 0; ri < sizeof rhos / sizeof rhos[0]; ri++)
            for (int rep = 0; rep < 60; rep++) {
                world_t w = {rhos[ri], 0.5, 0.08, 0.08};
                itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
                for (unsigned li = 0; li < sizeof lens / sizeof lens[0]; li++) {
                    double t[256], a[256], b; int fr[256], nn;
                    make_utt(&w, (int)(lens[li] * 450 / 84.0 + 0.5), t, a, fr, &nn, 1.0 + w.sent_sd * (2 * urand() - 1), &b);
                    itofs_plan_t p; itofs_sched_plan(&cal, lens[li], 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, NULL, 0, &p);
                    n++; nb += itofs_sched_simulate(t, a, nn, b, p.start_delay_us, NBUF, NULL) > 0;
                }
            }
        printf("jitter +-8%% and sentence cost +-8%% (outside the 25%% margin in the worst case): %ld/%ld sentences with an underrun (information, not a guarantee)\n", nb, n);
    }
    // ---- 4. the delay is not wasteful: the first audio is released as soon as the bound allows (never later than needed + 1 ms)
    {
        world_t w = {0.40, 0.5, 0.0, 0.0};
        itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
        itofs_plan_t p; itofs_sched_plan(&cal, 84, 0, NBUF, 1.0, 4e6, NULL, 0, &p);
        const int ok = p.start_delay_us - p.first_audio_us < 1000.0 && p.feasible;
        printf("rho 0.40: first chunk ready at %.0f ms, start delay %.0f ms (no extra wait needed) %s\n", p.first_audio_us / 1e3, p.start_delay_us / 1e3, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    // ---- 5. a board slower than real time cannot be fixed by waiting: reported, not hidden
    {
        world_t w = {1.15, 0.5, 0.0, 0.0};
        itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
        itofs_plan_t p; itofs_sched_plan(&cal, 175, 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, NULL, 0, &p);
        const int ok = !p.feasible && p.underruns > 0 && p.rtf > 1.0;
        printf("rho 1.15, 175 tokens: feasible %d, predicted underruns %d, delay %.0f ms (capped %.0f) %s\n", p.feasible, p.underruns, p.start_delay_us / 1e3, 4000.0, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    // ---- 6. just under real time (rho 0.97): still gapless with a long delay for a short sentence? (not promised for accepted sets only; reported)
    {
        long n_bad = 0, n = 0;
        for (int rep = 0; rep < 200; rep++) {
            world_t w = {0.93, 0.5, 0.02, 0.02};
            itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
            double t[256], a[256], b; int fr[256], nn;
            make_utt(&w, 450 * 40 / 84, t, a, fr, &nn, 1.0 + w.sent_sd * (2 * urand() - 1), &b);
            itofs_plan_t p; itofs_sched_plan(&cal, 40, 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, NULL, 0, &p);
            n++; n_bad += itofs_sched_simulate(t, a, nn, b, p.start_delay_us, NBUF, NULL) > 0;
        }
        printf("marginal board rho 0.93 (+-2%% jitter): %ld/%ld sentences with an underrun (information, not a guarantee)\n", n_bad, n);
    }
    // ---- 7. observed times replace the model: a board that turns out slower than calibrated raises the delay while waiting
    {
        world_t w = {0.60, 0.5, 0.0, 0.0};
        itofs_cal_t cal; calibrate(&w, 450, &cal, 84);
        itofs_plan_t p0, p1; itofs_sched_plan(&cal, 84, 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, NULL, 0, &p0);
        double obs[3] = {cal.t_us[0] * 1.5, cal.t_us[1] * 1.5, cal.t_us[2] * 1.5};
        itofs_sched_plan(&cal, 84, 0, NBUF, ITOFS_SCHED_SAFETY, 4e6, obs, 3, &p1);
        const int ok = p1.start_delay_us > p0.start_delay_us;
        printf("observed first chunks 50%% slower than calibrated: start delay %.0f -> %.0f ms %s\n", p0.start_delay_us / 1e3, p1.start_delay_us / 1e3, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
