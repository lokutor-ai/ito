# Benchmarking Ito on the ESP32-S3

This page does two things: it says what the published speed numbers are and what they depend on, and it proposes a protocol so that numbers from different boards, configurations and commits can be compared. The protocol is a proposal; the published numbers were not all taken with it (see "What the published numbers are").

## What the published numbers are

The root README and [`README.md`](README.md) §4a report, with the full split, for one ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), firmware `fedbd26`:

- first audio chunk (125 ms of audio) ready 184 ms after the phoneme ids reach the engine;
- RTF 0.66 (full split) on the boot calibration sentence (0.660 to 0.674 on the three demo sentences);
- planned gap-free start delay about 246 ms, 0 underruns.

Read them with these qualifications:

1. **They are for the full split, which is no longer the default.** Both cores run the GEMM (`s3_gemm_dual = 1`). The default build now uses the row split (the GEMM on core 0, the row loops on both cores), because the full split collapses the 3.3 V rail on some boards ([`DUAL_CORE_POWER.md`](DUAL_CORE_POWER.md)). The 0.66 is what you get after opting in on a board whose supply passes the stress test; the output audio is the same either way. A row-split figure from a different board is in the next section.
2. **They depend on the chunk plan.** The 0.66 uses the firmware's ramp of 125 ms, then 11, 12, 14, 18 and 24-frame chunks. Every chunk streams the layers' weights from PSRAM again, so the chunk size sets the weight traffic, and with it the RTF. An independent run on another board (below) went from RTF 1.06 at 8 frames per chunk to 0.84 at 24. A number without its chunk plan cannot be compared.
3. **They are one board, resets only, three sentences.** The repeat boots show repeatability on one chip, not a second board; none was a cold power-on; the spread reported is min to max over a handful of runs.
4. **The pre-board estimate columns (optimistic / central / pessimistic) are estimates**, are labelled so, and were not changed by the measurements.
5. **The int4 and light sets were timed by hand once** (female) or in one or two passes (male).

No published number in this repository comes from the one-core-GEMM weight stream bug fixed in `118a238` (the staged weight modes with `s3_gemm_dual = 0` split the stream across both cores while only core 0 ran the GEMM, which skipped half the output channels): that code path did not exist when the numbers were taken, and the output hash check would have flagged it. A configuration whose output hash differs from the host reference is not a valid row in any table.

### A second board, row split

The same Ito kernels (`s3_qgemm`, `s3_par`) driven by a separate harness on another ESP32-S3-WROOM-1 N16R8-class development board where the full split hangs. Female main int8 set, a 5.588 s sentence (134,100 samples at 24 kHz), 240 MHz, PSRAM at 80 MHz, flash QIO at 80 MHz, 64 KB data cache with 64-byte lines, 32 KB instruction cache, activations in internal SRAM, direct (cached) weight reads, every output bit-identical to the host reference. RTF is compute time over audio time; first chunk is the time to the first PCM.

| Configuration | Steady chunk (frames of 12.5 ms) | RTF | First chunk |
|---|---|---|---|
| One core | 8 | 1.323 | 298 ms |
| Row split | 4 | 1.425 | 206 ms |
| Row split | 8 | 1.056 | 246 ms |
| Row split | 12 | 0.932 | 278 ms |
| Row split | 16 | 0.903 | 310 ms |
| Row split | 24 | 0.836 | 380 ms |
| Row split, first chunk of 2 frames | 16 | 0.904 | 174 ms |
| Row split, first chunk of 8 frames | 20 | 0.862 | 220 ms |
| Row split, first chunk of 10 frames | 24 | 0.834 | 239 ms |
| Row split, weights copied to SRAM in tiles | 8 | 1.104 | 256 ms |
| Row split, weights moved by GDMA | 8 | 1.250 | 286 ms |
| Row split, activations in PSRAM | 8 | 1.205 | 283 ms |
| Row split, 160 MHz | 8 | 1.547 | 365 ms |
| Row split, PSRAM at 40 MHz | 8 | 1.380 | 320 ms |

What it says:

- The first chunk costs about 160 ms plus 7 ms per frame, so a small first chunk lowers the time to first audio without changing the steady-state RTF.
- Weights read per utterance (all weights are streamed once per chunk): 311.7 MB at 4 frames, 157.8 MB at 8, 106.4 MB at 12, 93.1 MB at 16, 64.3 MB at 24. The RTF follows it.
- Per-stage profile (one core, 7.34 s for the utterance): the int8 GEMMs are 3.41 s (46 %), 1.46 G MACs in 5,381 calls, so 0.43 GMAC/s or 1.8 MAC per cycle. The same kernel with its weights already in SRAM runs 0.79 GMAC/s. About 1.9 s of the 3.4 s of GEMM is cache-refill stall on PSRAM at 80 MHz (halving the PSRAM clock adds 1.87 s). GELU is about 0.47 s of the block time on one core.
- With the GEMM on one core (the row split), core 1 is busy about a third of the time; the loops split across cores are balanced to 1.5 %.
- GDMA staging (36 MB/s) is slower than the data cache (89 MB/s), so direct reads win, as in the published board run.

These are not comparable one to one with the 0.66 above: a different board, a different harness, the GEMM on one core, a different chunk plan and a different first-chunk definition. They show how far a board that cannot run the full split is from the published figure, and that the chunk size matters.

## Proposed protocol

Goal: any two numbers produced by it can be compared, and any number can be reproduced from its record.

### Metrics

| Metric | Definition |
|---|---|
| RTF | compute time of the whole utterance over the audio duration, from the CPU cycle counter, chunk plan fixed and recorded, below 1 is faster than real time |
| Time to first chunk | phoneme ids handed to the engine until the first chunk is computed and in memory |
| Time to first audio | the same plus the output stage, measured at the DAC or I2S end (needs a DAC or a logic analyser; otherwise report "not measured") |
| Gapless start delay | the delay needed for 0 underruns against the playout clock |
| Peak PSRAM and peak internal SRAM | high-water marks after the run (`heap_caps_get_minimum_free_size` or the engine's own counters), plus the hot arena size |
| Bit-exactness | hash (fnv32 or sha256) of all PCM against the host reference build of the same commit, same weights, same chunk plan; the row is invalid if it differs |
| Weight traffic | bytes read from PSRAM per utterance, so that two chunk plans can be compared |

### Fixed inputs

- Weights: the female main int8 set, then the male main int8 set (and the int4 and light sets as separate runs). Record the file hashes.
- Sentences: the three demo sentences (25, 108 and 175 tokens) plus the 84-token boot calibration sentence, as phoneme ids from the host front end, identical across runs. Add one short (under 2 s) and one long (over 12 s) sentence to see the ramp and the steady state.
- Chunk plan: the firmware's default plan for the headline row (the 10-frame first chunk, then 11, 12, 14, 18 and 24 frames), and 8, 16 and 24 fixed frames for the chunk-size rows.
- Configuration: the row split (the default) and the full split (opt-in) as separate rows. A board whose full split fails the stress test in [`DUAL_CORE_POWER.md`](DUAL_CORE_POWER.md) reports the row split only and says so.

### Procedure

1. Cold start: power the board off and on, wait for boot. Record the reset reason (a power-on reset, not a USB or watchdog reset).
2. Warm-up: run each sentence once and discard it (cache, PSRAM and arena warm).
3. Measure at least 10 repetitions per sentence and configuration, back to back, in a fixed order, with nothing else running (no WiFi, no logging in the loop, I2S idle or active as stated).
4. Check the output hash after every repetition. Any difference stops the run and is reported.
5. Report the mean, the standard deviation and the minimum and maximum over repetitions, and the number of repetitions. Repeat the whole procedure after at least two more cold starts and report the spread between boots as well.
6. Repeat on at least a second board of the same module class when claiming anything about "the ESP32-S3".

### Configuration dump

Every table carries, or links to, the following, printed by the firmware at boot or read from the build:

- chip, revision, module (flash and PSRAM size and mode), CPU, flash and PSRAM clocks and modes (for example 240 MHz, QIO 80 MHz, octal 80 MHz);
- instruction and data cache sizes, ways and line sizes;
- ESP-IDF version, compiler version, optimisation flags, the sdkconfig hash;
- Ito commit, weight file hashes, chunk plan, split mode (`s3_dual_enabled`, `s3_gemm_dual`), weight mode (`s3_wmode`), activation width, tile size;
- the supply: power source, cable, whether the brownout detector fired during the run (zero events is part of a valid run);
- the stress-test result of the board ([`DUAL_CORE_POWER.md`](DUAL_CORE_POWER.md)).

### Host reference

Run the same sentences with the same weights through the host build (`esp32/host`, `ito_cli`) and record its PCM hash and, for context, its RTF. Every board row must carry the same hash. The host build is bit-identical to the board by design, so it is the reference for correctness, not for speed.

### Table format

One row per (configuration, chunk plan, sentence), one header with the configuration dump reference:

| Board id | Commit | Split | Weights | Chunk plan | Sentence | RTF mean (sd) [min..max] | First chunk ms mean (sd) | First audio ms | Peak PSRAM / SRAM KB | n | Hash | Boots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Rules for the table: show the hash column and mark rows that match the host reference; never mix chunk plans in one row; list estimates in a separate table from measurements; put "not measured" instead of a blank.

### What the protocol would change in the published numbers

- The headline would carry its chunk plan and split.
- The spread would be over cold boots and repetitions, not a single boot.
- Peak memory would be given per run, and the hot arena size alongside it.
- A second board would be required for any claim about the chip in general.

### Kernel check and cycle benchmark

`firmware/main/kernel_check_s3.c` (build with `-DITOFS_KCHECK`, call `s3_kernel_check(1)`) compares every GEMM kernel variant in its table with a plain C int32 reference on random and extreme-valued data (972 shapes per variant: 1 to 24 rows, 1 to 76 vectors, 1 to 33 weight rows), then, on a chip, measures CPU cycles per MAC with the weights in internal SRAM and streamed from PSRAM. It is a way to judge a kernel change without running the synthesiser; QEMU runs the exactness part only.

Findings from experiments on one board (cycles per MAC, 8 activation rows, 240 MHz):

- The dot kernel costs about 21 cycles per weight row for 1 or 2 vectors (a fixed cost) and roughly 3 cycles per vector beyond that, near the dependent-accumulate rate of ACCX. Unrolling the loop 4 times made it slower (0.96 against 0.67 cycles per MAC at 32 inputs). A version that alternates two register pairs and issues the next row's loads before reading ACCX was equal at 64 inputs and slower below it (0.81 against 0.67 at 32).
- Starting the data cache preload engine (`EXTMEM_DCACHE_PRELOAD_*`) on PSRAM weights hung the board at its first use in this form; it is not used.
