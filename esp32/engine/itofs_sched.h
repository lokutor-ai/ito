// ItoFS boot calibration and playback scheduling: the policy that turns MEASURED per-chunk times into (1) a choice between weight sets and
// (2) the playback start delay that keeps the speaker from running dry. Portable C99 with no timing source of its own: the firmware feeds it
// microsecond measurements, and the host / QEMU tests feed it simulated ones (host/host_test_sched.c, the firmware's `simtime` command).
//
// What this can and cannot promise. If the chunk times measured at boot are representative of later sentences, then for a weight set with a steady
// real-time factor below 1 the delay computed here makes playback gapless (the test simulates the speaker chunk by chunk). It cannot make a
// board that is slower than real time fast: it then reports `degraded` instead of silently stuttering. It is a guarantee BY MEASUREMENT on the
// board that runs it, not a statement about silicon nobody has measured yet.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITOFS_SCHED_MAXCH 64          // chunks of a calibration trace
#define ITOFS_SCHED_MAXSETS 4         // weight sets the ladder can hold
#define ITOFS_SCHED_PLANCH 48         // chunks a plan models (the start delay is set by the ramp and the first chunks; later chunks only add margin when RTF < 1)

// ---- thresholds (the owner's decisions) ------------------------------------------------------------------------------------
#define ITOFS_RTF_TARGET 0.85         // a weight set is accepted when its measured RTF is at or below this
#define ITOFS_RTF_DEGRADED 0.95       // the best set still at or above this: warn and raise the degraded flag
#define ITOFS_SCHED_SAFETY 1.25       // measured chunk times are scaled by this before they are trusted for the start delay (jitter margin)

// A calibration trace: one representative utterance synthesised without playback, chunk by chunk, with the schedule the player will use.
typedef struct {
    int n;                            // chunks measured
    int hop, sr;                      // samples per frame, sample rate
    int frames[ITOFS_SCHED_MAXCH];    // frames in chunk k
    double t_us[ITOFS_SCHED_MAXCH];   // wall time to produce chunk k (compute plus weight fetch), us
    double begin_us;                  // wall time of the utterance set-up before chunk 0 (text side), us
    int steady_frames;                // frames of a full steady-state chunk (24)
    int n_tok;                        // tokens of the calibration utterance
} itofs_cal_t;

// Steady-state real-time factor of a trace: the median over its full-size chunks of (production time / audio time), and the whole-trace figure
// (all chunks including the ramp, begin time excluded); the larger of the two is returned (the conservative one). *steady_us gets the median
// full-chunk production time. Returns < 0 if the trace has no full-size chunk.
double itofs_cal_rtf(const itofs_cal_t *c, double *steady_us, double *whole);

typedef enum { ITOFS_TIER_OK = 0, ITOFS_TIER_MARGINAL = 1, ITOFS_TIER_DEGRADED = 2 } itofs_tier_status_t;

// Choose among weight sets in order of preference (index 0 = most preferred, the best quality): the first whose measured RTF is <= target
// (default ITOFS_RTF_TARGET). If none is, the one with the lowest RTF; its status is MARGINAL (target < rtf < degraded) or DEGRADED
// (rtf >= degraded, default ITOFS_RTF_DEGRADED). rtf[i] < 0 means "not measured / unavailable" and is skipped. Returns the index or -1.
int itofs_sched_pick(const double *rtf, int n, double target, double degraded, itofs_tier_status_t *status);

// Playback plan for one utterance of n_tok tokens, from a calibration trace of the SAME weight set and schedule.
typedef struct {
    double start_delay_us;            // hold playback until this long after the text arrived (>= the first chunk's production time)
    double first_audio_us;            // production time of chunk 0 (the earliest the speaker could start)
    int underruns;                    // underruns the model predicts with that delay (0 unless infeasible)
    int feasible;                     // 1 if some delay gives no underrun within the cap
    int nchunks;                      // chunks modelled
    double rtf;                       // steady RTF used
} itofs_plan_t;

// nbuf: PCM chunk buffers between synthesis and the speaker (production of chunk k waits until the buffer of chunk k - nbuf has played).
// obs_t_us / n_obs: production times actually measured for the first n_obs chunks of THIS utterance (NULL / 0 when none yet); they replace the
// model for those chunks, which lets the player raise the delay while it waits. max_delay_us caps the delay (e.g. 4 s).
// Chunk times for chunks past the calibration trace use its steady median; the frames of chunk k are calib->frames[k] while k < n, else steady_frames.
// The number of chunks is estimated from n_tok with the trace's frames per token (n_frames_hint > 0 overrides it).
void itofs_sched_plan(const itofs_cal_t *cal, int n_tok, int n_frames_hint, int nbuf, double safety, double max_delay_us,
                      const double *obs_t_us, int n_obs, itofs_plan_t *out);

// Discrete-event model of the speaker for tests and for the plan itself: with chunk production times t_us[] (serial synthesis, one buffer pool of nbuf),
// audio durations a_us[] and a start delay, returns the number of underruns (chunks whose audio is not ready when the previous chunk ends) and
// the time playback finishes. begin_us is added before chunk 0. Production of chunk k cannot start before chunk k - nbuf has started playing + its duration.
int itofs_sched_simulate(const double *t_us, const double *a_us, int n, double begin_us, double start_delay_us, int nbuf, double *end_us);

#ifdef __cplusplus
}
#endif
