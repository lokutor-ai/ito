// ItoFS portable C99 inference engine (FastSpeech-style front + Vocos-style frame-rate decoder with a harmonic source).
//
//   tokens --emb--> conv encoder --> BiGRU --> duration head (style FiLM) --> durations (frames)    [itofs_begin, once]
//   length-regulated features --> prosody net (style FiLM) --> log-F0, voicing, energy               [frame rate, streamed]
//   harmonic source (8 sines at F0, phase accumulator, noise) --> STFT --> harm_proj
//   arch 2 (v2):  [features | curves] --> embed (+ style cond) --> ConvNeXt blocks --> log-mag / phase --> iSTFT --> PCM
//   arch 3 (v3):  [features | curves] --> mel head (1x1, ConvLN x L with style FiLM, 1x1) --> 100-bin log-mel
//                 [log-mel | vocoder log-F0 | voiced] --> embed (no style) --> ConvNeXt blocks --> ... --> iSTFT --> PCM
//   The blob header key `arch` (default 2) selects the graph; every dimension is read from the blob.
//
// Numerics (identical to the engine-numerics reference used at export, modes w8a8 (default) / w8a16):
//   * every dense Conv1d / Linear on the token and frame paths has integer weights, per output channel. The F0 path
//     (text encoder, GRU input matrices, rnn_proj, duration and prosody convs) has int16 weights stored as two int8
//     planes (q16 = 256*hi + lo) and 15-bit activations (q = floor(x * 16256 / max|x_t| + 0.5) = 128*hi + lo):
//     4 exact int8 GEMMs. The decoder has int8 weights and, by default, ONE int8 activation plane per frame
//     (q = floor(x * 127 / max|x_t| + 0.5)); act_bits = 16 gives it the 15-bit two-plane path instead.
//     Activations are quantised PER ROW (token or frame); int32 accumulators are exact and rescaled in float.
//   * float32: embeddings, style projection, FiLM, GRU recurrence, LayerNorm, depthwise convs, GELU, the output heads of
//     the duration and prosody nets, the harmonic source, STFT / iSTFT (own mixed-radix FFT: n_fft 1200 / 2400).
//   * transcendental functions are the engine's own, built from IEEE + - * / only (compile with -ffp-contract=off),
//     so the host and the chip produce bit-identical PCM.
//   Per-row activation scales make a frame's result independent of which frames share its batch, so streamed output is
//   BIT-IDENTICAL to whole-utterance output for every chunk size.
//
// Streaming: the text side runs once in itofs_begin() (the sentence is known). The frame side is a chain of stages
// (prosody layers, curves, source, harmonic features, embed, blocks, head, overlap-add), each with a small ring buffer
// and its exact receptive field; each itofs_next_chunk() computes every stage only as far as the requested audio needs.
//
// Memory: the weight blob is read in place. Every other buffer comes from two caller arenas (hot scratch + bulk state)
// whose sizes itofs_arena_bytes() gives up front; nothing is allocated after itofs_init().
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITOFS_MAX_LAYERS 8     // text / duration / prosody conv layers
#define ITOFS_MAX_BLOCKS 16    // decoder ConvNeXt blocks

typedef struct {               // int8 or int16 conv / linear, tap-major weights [K][out][in]
    const int8_t *w;           // int8 weights, or the high plane of int16 weights
    const int8_t *w_lo;        // NULL for int8; low plane of int16 weights (q16 = 256 * w + w_lo, |q16| <= 32639)
    const float *sw;           // [out] per-output-channel scale
    const float *b;            // [out] or NULL
    int K, out, in;
    int act16;                 // int8-weight layer that always takes 15-bit (two-plane) activations (blob entry flag bit 0)
} itofs_qlin_t;

typedef struct {               // ConvLN: conv -> LayerNorm -> (FiLM) -> GELU, residual
    itofs_qlin_t conv;
    const float *ln_g, *ln_b;
    const float *film_w, *film_b;   // [2*dim][style_dim], [2*dim] or NULL
    itofs_qlin_t film_q;            // the FiLM linear with integer weights (film_q.w != NULL instead of film_w)
} itofs_convln_t;

typedef struct {
    const float *dw_w, *dw_b;  // depthwise [C][K], [C] (torch layout [C,1,K])
    const float *ln_g, *ln_b, *gamma;
    itofs_qlin_t pw1, pw2;
} itofs_block_t;

typedef struct {
    // config (blob header)
    int n_vocab, text_dim, text_layers, text_kernel, text_rpad, rnn_hidden, rnn_bidir, style_in, style_dim;
    int dur_layers, dur_kernel, dur_rpad, sent_pos, hf_dim, pros_dim, pros_layers, pros_kernel, pros_rpad, n_pros_out;
    int dec_in, dec_dim, embed_kernel, dec_inter, dec_blocks, dec_kernel, dec_rpad, harm_in, dec_out_dim;
    int n_fft, hop, sr, n_harm, fps, n_styles;
    int arch;                  // 2: v2 (decoder on text features + curves), 3: v3 (mel head + mel vocoder)
    int n_mels, mel_dim, mel_layers, mel_kernel, mel_rpad;     // arch 3
    float src_amp, src_noise, uv_hz, mag_max, eps_text, eps_dec;
    float voc_hz;              // arch 3: vocoder voicing threshold (Hz) applied to the front's F0 (Vocoder.clean_f0)
    const float *stats;        // lf0 mean, lf0 std, energy mean, energy std
    const float *f0stats;      // arch 3: vocoder log-F0 mean, std
    const float *mel_mean, *mel_std;                           // arch 3: [n_mels]
    // weights
    const float *emb;          // [n_vocab][text_dim]
    const float *style_bank;   // [n_styles][style_in] (NULL when the blob has a style table)
    const float *style_table;  // optional [n_styles][style_table_w]: per style, the FiLM (1 + g, b) of every duration, prosody
                               // and mel-head layer, precomputed by the exporter (then no style bank / style_proj / FiLM
                               // weights are stored: ~0.8 MB less, no per-utterance FiLM work)
    int style_table_w;
    const float *style_w, *style_b;
    itofs_qlin_t style_q;      // style_proj with integer weights (instead of style_w)
    // text -> style predictor (blobs with "sp.*"; replaces the style bank): style = mu + MLP([mean_t h, max_t h, log(n)/5]),
    // h = the style-free text encoder output (encoder + GRU + rnn_proj) of the WHOLE sentence; runs before any duration
    int has_sp;
    itofs_qlin_t sp[3];        // Linear(2*text_dim+1 -> hid), GELU, Linear(hid -> hid), GELU, Linear(hid -> style_in)
    const float *sp_mu;        // [style_in]
    itofs_convln_t enc[ITOFS_MAX_LAYERS], dur[ITOFS_MAX_LAYERS], pros[ITOFS_MAX_LAYERS];
    itofs_qlin_t rnn_ih[2], rnn_proj;          // GRU input matrices (bias = b_ih; [1] only if bidirectional), output projection
    const float *rnn_hh[2], *rnn_bhh[2];       // [3H][H], [3H] (gates r, z, n)
    const float *dur_out_w, *dur_out_b;        // [1][text_dim]
    itofs_qlin_t pros_in;
    const float *pros_out_w, *pros_out_b;      // [3][pros_dim]
    itofs_qlin_t mel_in, mel_out;              // arch 3 mel head: 1x1 in, ConvLN layers, 1x1 out
    itofs_convln_t mel[ITOFS_MAX_LAYERS];
    itofs_qlin_t dec_embed, harm_proj, dec_out;
    const float *cond_w, *cond_b, *n0_g, *n0_b, *n1_g, *n1_b;
    itofs_block_t blk[ITOFS_MAX_BLOCKS];
    size_t blob_size;
    long n_int8_params, n_int16_params, n_f32_params;
} itofs_model_t;

// Parse a blob (must stay alive, 16-byte aligned). Returns 0 or a negative error.
int itofs_model_init(itofs_model_t *m, const void *blob, size_t size);
// Find any tensor in a blob by name (e.g. the optional int32 "selftest" record). Returns 0 or a negative error.
int itofs_blob_find(const void *blob, size_t size, const char *name, const void **data, size_t *nbytes, int *dtype);

// ---- kernels the host can override --------------------------------------------------------------------------------
// acc[r][o] = sum_{i<in} x[r][i] * w[o][i]  (x rows stride ldx, a multiple of 16 with 16-byte aligned rows, and the bytes of
// every row between `in` and ldx are ZERO, so a kernel may run whole 16-byte vectors over them; w rows stride `in`, any
// alignment; acc rows stride `out`), exact int32.
typedef void (*itofs_qgemm_fn)(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user);
void itofs_qgemm_ref(const int8_t *x, int rows, int ldx, int in, const int8_t *w, int out, int32_t *acc, void *user);

// Debug taps: called with every frame of an intermediate.
typedef void (*itofs_tap_fn)(void *user, int what, int t, const float *row, int width);
enum { ITOFS_TAP_PRED = 1, ITOFS_TAP_SRC = 2, ITOFS_TAP_HFEAT = 3, ITOFS_TAP_SPEC = 4,
       ITOFS_TAP_MEL = 5,      // arch 3: log-mel row (n_mels)
       ITOFS_TAP_F0 = 6 };     // source F0 of the frame (Hz, 0 = unvoiced; 1 value)

// MAC accounting (logical multiply-accumulates; the two activation planes count once)
enum { ITOFS_G_TEXT, ITOFS_G_PROS, ITOFS_G_HARM, ITOFS_G_EMBED, ITOFS_G_BLOCKS, ITOFS_G_HEAD, ITOFS_G_MEL, ITOFS_G_N };

// ---- context --------------------------------------------------------------------------------------------------------
typedef struct {
    int max_tokens;            // longest token sequence itofs_begin accepts (the corpus uses <= 400)
    int chunk_frames;          // frames per itofs_next_chunk step; 1 frame = hop samples
    int act_bits;              // activations of the int8-weight layers: 8 (one int8 plane, default when 0) or 16 (two
                               // planes). Layers with int16 weights always take 16-bit activations. Can be changed in
                               // the context (c->act_bits) between utterances.
    int whole_text;            // 1: also allocate the whole-sentence text-side buffers when the model does not need them
                               // (incremental-capable models; lets c->force_whole_text compare both paths, ~1.2 MB)
} itofs_limits_t;

// Optional dual-core row parallelism (the chip's second core): par(user, body, arg, n) must call body(arg, i0, i1, core)
// over a partition of [0, n) -- each part exactly once, core 0 or 1 -- and return when all parts are done. Every body
// writes disjoint rows with per-core scratch, so the result does not depend on the partition (bit-identical).
typedef void (*itofs_body_fn)(void *arg, int i0, int i1, int core);
typedef void (*itofs_par_fn)(void *user, itofs_body_fn body, void *arg, int n);

typedef struct itofs_stage itofs_stage_t;
typedef struct itofs_tstage itofs_tstage_t;
typedef struct itofs_fft itofs_fft_t;

typedef struct {
    const itofs_model_t *m;
    itofs_limits_t lim;
    itofs_qgemm_fn qgemm; void *qgemm_user;
    itofs_tap_fn tap; void *tap_user;
    itofs_par_fn par; void *par_user;              // NULL: everything on the calling core (taps disable it)
    // ---- test hooks (NULL / 0 in normal use) ----
    const float *force_h;      // [n][text_dim]: replaces the text encoder output (after the GRU)
    const int *force_dur;      // [n]: replaces the predicted durations
    const float *force_pred;   // [T][3]: replaces the prosody net output (normalised log-F0, voicing logit, energy)
    const float *force_f0;     // [T]: replaces the source F0 in Hz (0 = unvoiced) derived from the prosody output
    const float *force_mel;    // arch 3, [T][n_mels]: replaces the mel head output
    const float *ext_noise;    // [T*hop]: source noise instead of the seeded generator
    double ext_phase0; int use_ext_phase0;
    int force_whole_text;      // 1: run the whole-sentence text side even when the incremental one is possible
    int text_step;             // incremental text side: tokens added per step (0 = chunk_frames; 1 = least work before
                               // the first chunk, but GEMMs on single rows lose the weight reuse)
    // ---- per utterance ----
    // T: frames of the utterance once known. With the INCREMENTAL text side (forward-only GRU, no sent_pos) the text
    // side runs token by token as the frame side needs it, and T is known only when the last token's duration is
    // (T_known = 1); until then T holds a large sentinel and only frames < start[n_done] are used.
    int n_tok, T, T_known, produced;
    int text_incr, n_done;     // incremental text side active; tokens whose duration (and start) are final
    uint32_t seed;
    double phase0, src_ph;     // source phase (cycles) after the last generated sample
    int *dur, *start;          // [max_tokens (+1)]
    float *h, *logd;           // [max_tokens][text_dim], [max_tokens]
    float *s, *cond;           // [style_dim], [dec_dim]
    float *film_g1[ITOFS_MAX_LAYERS], *film_b[ITOFS_MAX_LAYERS];   // pros layers: (1 + g), b
    float *dfilm_g1[ITOFS_MAX_LAYERS], *dfilm_b[ITOFS_MAX_LAYERS]; // duration layers: (1 + g), b
    float *mfilm_g1[ITOFS_MAX_LAYERS], *mfilm_b[ITOFS_MAX_LAYERS]; // arch 3 mel head layers: (1 + g), b
    int *tok;                  // [max_tokens] copy of the utterance's tokens
    float *gru_h;              // forward GRU state (incremental text side)
    itofs_tstage_t *tst; int n_tst, tring, tring_mask, text_la;   // token stages; text_la = token lookahead
    double macs[ITOFS_G_N], macs_f32[ITOFS_G_N];   // accumulated since itofs_init: logical MACs (int GEMM, float dense)
    double macs_exec[ITOFS_G_N];                   // int8 x int8 MACs the GEMM kernel actually executes (x planes)
    int act_bits;              // 8 or 16 (see itofs_limits_t), read by itofs_begin
    int act_planes;            // 1 or 2 for the current utterance
    // ---- internals ----
    int nb, ring, ring_mask, n_stages;
    itofs_stage_t *st;
    itofs_fft_t *fft;
    int ola_L, ola_R, ola_off;
    float *out_ptr; int out_base;
    float *t_x, *t_y, *t_g, *t_o, *t_pad, *t_film;           // text-side buffers
    float *s_g0, *s_g1, *s_y0, *s_y1, *s_qs, *s_pcm;  // per-step scratch
    int8_t *s_q8; int32_t *s_acc;
    float *win, *win2, *fbuf, *rtw;
    float *fbuf_b, *x_b;       // second core's FFT buffer and STFT frame / half spectrum (row-parallel HFT and HEAD)
    itofs_fft_t *fft_b;        // second core's FFT plan (shares the twiddles, own scratch)
    double *src_anc;           // source phase anchors of a batch (2 per frame)
    float *src_w;              // [hop]: (j + 0.5) / hop, the source's linear-interpolation weights
    int acc_tile, tb;          // tb: whole-sentence text-side batch (tokens per GEMM call)
    long cap_g0, cap_y0, cap_y1, cap_q8, cap_qs;   // hot scratch capacities (floats; bytes for q8; rows for qs):
                               // a stage may run more than chunk_frames rows at once when they fit (the first chunk's
                               // pipeline fill then reads each weight matrix once instead of once per chunk_frames)
    size_t acc_bytes;
    size_t hot_used, bulk_used;
} itofs_ctx_t;

// Two arenas: `hot` = per-step scratch (internal SRAM on the chip), `bulk` = text-side buffers and stage rings (PSRAM).
void itofs_arena_bytes(const itofs_model_t *m, const itofs_limits_t *lim, size_t *hot_bytes, size_t *bulk_bytes);
int itofs_init(itofs_ctx_t *c, const itofs_model_t *m, const itofs_limits_t *lim,
               void *hot, size_t hot_bytes, void *bulk, size_t bulk_bytes);

// Start an utterance: tokens are StyleTTS 2 symbol ids with the leading 0 pad; style_idx picks a bank entry; seed fixes
// the source noise and start phase (the same seed gives the same audio for any chunking). Returns T (frames) or <0.
int itofs_begin(itofs_ctx_t *c, const int *tokens, int n, int style_idx, uint32_t seed);

// Next chunk: up to min(chunk_frames, max_samples / hop) frames. Returns samples written (a multiple of hop),
// 0 when the utterance is finished, <0 on error.
int itofs_next_chunk_f32(itofs_ctx_t *c, float *out, int max_samples);
int itofs_next_chunk(itofs_ctx_t *c, int16_t *out_pcm, int max_samples);   // float -> int16 (x32767, round, clamp)

const char *itofs_strerror(int err);

#ifdef ITOFS_OPCOUNT   // host profiling build: counts of the non-GEMM float work (tools: engine/opcount_v3.c)
enum { ITOFS_OPC_EXP, ITOFS_OPC_LOG, ITOFS_OPC_SINCOS, ITOFS_OPC_ERF, ITOFS_OPC_ERF_SERIES, ITOFS_OPC_QUANT, ITOFS_OPC_LN,
       ITOFS_OPC_FFT, ITOFS_OPC_GAUSS, ITOFS_OPC_SRC, ITOFS_OPC_N };
extern long long itofs_opc[ITOFS_OPC_N];
#endif

#ifdef ITOFS_PROF      // profiling build: time per frame-stage kind (PIN PROS CUR SRC HFT EMB BLK HEAD OLA MIN MEL MOUT),
                      // token-stage kind (ENC GRU PROJ DUR DOUT), and inside the dense layers: all, quantise, GEMM calls
enum { ITOFS_PROF_STAGE = 0, ITOFS_PROF_TSTAGE = 12, ITOFS_PROF_QLIN = 17, ITOFS_PROF_QUANT, ITOFS_PROF_GEMM,
       ITOFS_PROF_SGEMM = 20,           // GEMM-call time inside each frame-stage kind (12) and token-stage kind (5)
       ITOFS_PROF_P0 = 37,              // inside the stages: block depthwise+LN, block GELU, block residual, rescale (all qlin), bias (all qlin),
                                        // gather/put_rows, row LayerNorms (EMB, HEAD), head_body, hft_body, src_body+anchors, ola_body, cln_body, fused rescale (8-bit path)
       ITOFS_PROF_N = 37 + 13 };
extern uint32_t (*itofs_prof_clock)(void);
extern double itofs_prof[ITOFS_PROF_N];
void itofs_prof_micro(double *res);   // CCOUNT ticks per call: div, GELU (|x|<=1), GELU (tail), expf, sqrtf, sincosf, logf, mul+add
#endif

#ifdef __cplusplus
}
#endif
