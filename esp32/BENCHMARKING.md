# Benchmarking Ito on the ESP32-S3

This page does two things: it says what the published speed numbers are and what they depend on, and it proposes a protocol so that numbers from different boards, configurations and commits can be compared. The protocol is a proposal; the published numbers were not all taken with it (see "What the published numbers are").

## What the published numbers are

The root README and [`README.md`](README.md) §4a report, with the full split, for one ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), firmware `fedbd26`:

- first audio chunk (125 ms of audio) ready 184 ms after the phoneme ids reach the engine;
- RTF 0.66 (full split) on the boot calibration sentence (0.660 to 0.674 on the three demo sentences);
- planned gap-free start delay about 246 ms, 0 underruns.

Read them with these qualifications:

1. **They are for the full split.** Both cores run the GEMM (`s3_gemm_dual = 1`). The firmware now chooses the split at run time: it uses the full split on a board whose supply passes its self-test and the row split (the GEMM on core 0, the row loops on both cores) on one that does not, because the full split collapses the 3.3 V rail on some boards ([`DUAL_CORE_POWER.md`](DUAL_CORE_POWER.md)). The 0.66 is what a board whose test passes runs at; the output audio is the same either way. A row-split figure from a different board is in the next section, and the row split on the first board is in the section at the end.
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

### A second board, configuration sweep

One change per build on the same second board and harness, everything else as in the table above (female main int8 set, 5.59 s sentence, 240 MHz unless stated, PSRAM at 80 MHz, direct weight reads, hot arena in internal SRAM, every output bit-identical to the host reference). RTF for one core at 8 frames per chunk / row split at 8 / row split at 16. The baseline is 1.323 / 1.057 / 0.902.

| Change | RTF (one core, 8 / row split, 8 / row split, 16) |
|---|---|
| `-funroll-loops` | 1.309 / 1.060 / 0.905 |
| D-cache 32 KB instead of 64 KB | 1.326 / 1.061 / 0.905 |
| D-cache 4 ways instead of 8 | 1.323 / 1.057 / 0.902 |
| D-cache line 32 B instead of 64 B | 1.507 / 1.240 / 1.021 |
| I-cache 16 KB instead of 32 KB | 1.331 / 1.086 / 0.928 |
| CPU 160 MHz | 1.817 / 1.547 / 1.392 |
| PSRAM 40 MHz | 1.642 / 1.711 / 1.478 |

The PSRAM 40 MHz row split at 8 frames (1.711) comes from this sweep; the 1.380 in the table above is a different run of the same change.

Also measured on that board:

- **GDMA staging.** An 8 KB tile instead of 4 KB takes one-core GDMA mode from 1.520 to 1.372, still slower than direct reads (1.323). One-core memcpy to SRAM is 1.371 and one-core GDMA 1.516.
- **A 20-frame steady chunk after an 8-frame first chunk** gives RTF 0.862 and a 220 ms first chunk.
- **Check where the hot arena lands.** A hot arena that spills to PSRAM costs RTF 1.205 against 1.056 at chunk 8 (row split). Start the kernels' worker before reserving the arena, so that it gets internal SRAM.

Provenance of this section: the lokutor bench harness, a second board, female_main (5.59 s), 240 MHz unless stated, PSRAM at 80 MHz. The raw logs are not included in this repository; the numbers are copied from the harness notes.

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

## Re-measured on 2 November 2026 (ESP32-S3-DevKitC-1 N16R8, MAC f8:5b:1b:91:6d:a8, branch `production-runtime`)

Same protocol as above where practical: n = 10 repetitions, the PCM hash of every run compared with the host C engine, the three demo sentences and the 84-token calibration sentence, both splits, three weight sets, ESP-IDF 5.5.1, 240 MHz, PSRAM octal 80 MHz, 64 KB data cache with 64-byte lines. The board sits on an ESP_S3-AUDIN shield with both USB-C cables connected. Every boot was a reset; the supply of this board passed everything (stress program round of 422 s with 0 mismatches, the firmware's own 3 s test, hours of synthesis, speaker playing at +4.5 dB: 0 brownout and 0 watchdog resets).

### Headline: main as published against this branch (female voice, demo sentences, ramp 10,11,12,14,18,24, 0 underruns everywhere)

| Firmware | Split | Set | First chunk ms | RTF (calibration sentence / demos) | Planned start delay ms (25/108/175 tokens) | Peak SRAM KB | Hash |
|---|---|---|---|---|---|---|---|
| main 9c8105d, prebuilt image (IDF 5.5.5), n=10 | row | main int8 | 236.8 | 0.845 / 0.843-0.860 | 513 / 697 / 657 | 366 of 374 | self-test fnv32 023bd6b1 = host, 31 of 31 |
| same, `tier 1` / `tier 2` | row | int4 / light | 231.7 / 216.5 | 0.826 / 0.776 | not planned (the set was not calibrated by the boot) | 366 | self-test PASS 31 of 31 each |
| main 9c8105d built with the full split (IDF 5.5.1), n=10 | full | main int8 | 183.1 (183.0-184.0) | 0.661 / 0.660-0.674 | 246 / 246 / 246 | 354 of 374 | 023bd6b1 = host, 31 of 31 |
| same | full | int4 / light | 179.3 / 167.7 | 0.648 / 0.608 | not planned | 354 | PASS |
| this branch, `auto` (the stress-passing board runs the full split) | full | light (chosen by the new policy) | 167.0 | 0.608 / 0.607-0.615 | 167 / 167 / 167 | 346 of 371 | demo hashes = host, 10 of 10 |
| this branch, forced row | row | light (chosen) | 216.3 | 0.776 / 0.773-0.784 | 416 / 476 / 392 | 346 | = host |
| this branch, forced row, main set | row | main | 236.5 | 0.844 / 0.841-0.858 | 511 / 696 / 646 | 346 | = host |

First chunk is 125 ms of audio. The new set choice picks, among the sets that meet RTF 0.85 and are judged equivalent by ear, the lowest planned start delay: the light set, which cuts the start delay from 246 to 167 ms in the full split and from 511-697 to 392-476 ms in the row split. The three sets are indistinguishable by ear for both voices (Dani's blind tests); `SETS_EQUIVALENT` in `main.c` turns the choice back to the first set in quality order.

### Protocol table, branch firmware, female voice (n = 10 silent repetitions per cell with the PCM hash of every run compared with the host C engine, then 3 played runs for the planned start delay and underruns)

Modes `row` (forced), `full` (forced) and `auto` (verdict from the supply test; it ran the full split here, its rows are identical to `full` and listed once). The build of this run is the firmware of commit cea7be6 before the later additions (volume in percent, DAC mute, loopback, chunk/cores/trace commands, silent boot); the synthesis code is unchanged since.

| Board id | Commit | Split | Weights | Chunk plan | Sentence | RTF mean (sd) [min..max] | First chunk ms mean (sd) | Planned start delay ms / underruns (3 played runs) | Peak PSRAM / SRAM KB | n | Hash vs host | Boots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.673 (0.0005) [0.672..0.674] | 183.0 (0.19) | 249 / 0 | 5669 / 346 | 10 | a39e1a7b = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.659 (0.0004) [0.659..0.660] | 182.8 (0.33) | 249 / 0 | 5669 / 346 | 10 | f6afba59 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.662 (0.0002) [0.662..0.663] | 182.9 (0.25) | 249 / 0 | 5669 / 346 | 10 | 18154ad5 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.656 (0.0002) [0.655..0.656] | 179.1 (0.12) | 221 / 0 | 5669 / 346 | 10 | 0284b26f = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.646 (0.0002) [0.646..0.646] | 178.8 (0.17) | 221 / 0 | 5669 / 346 | 10 | 251a24a1 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.649 (0.0001) [0.649..0.650] | 178.8 (0.19) | 221 / 0 | 5669 / 346 | 10 | d18ddec4 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.615 (0.0003) [0.614..0.615] | 167.0 (0.09) | 167 / 0 | 5669 / 346 | 10 | 4961ecd4 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.607 (0.0002) [0.607..0.607] | 166.9 (0.18) | 167 / 0 | 5669 / 346 | 10 | 1be9e33a = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | full | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.610 (0.0001) [0.610..0.610] | 167.1 (0.13) | 167 / 0 | 5669 / 346 | 10 | 98faff55 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.858 (0.0001) [0.857..0.858] | 236.6 (0.06) | 511 / 0 | 5669 / 346 | 10 | a39e1a7b = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.841 (0.0001) [0.841..0.841] | 236.4 (0.07) | 696/697 / 0 | 5669 / 346 | 10 | f6afba59 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 0 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.844 (0.0001) [0.844..0.844] | 236.5 (0.05) | 646 / 0 | 5669 / 346 | 10 | 18154ad5 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.836 (0.0001) [0.836..0.837] | 231.6 (0.07) | 514/516 / 0 | 5669 / 346 | 10 | 0284b26f = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.823 (0.0001) [0.823..0.823] | 231.5 (0.06) | 616/618 / 0 | 5669 / 346 | 10 | 251a24a1 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 1 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.826 (0.0001) [0.826..0.827] | 231.5 (0.07) | 571/572 / 0 | 5669 / 346 | 10 | d18ddec4 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 0 (25 tok) | 0.784 (0.0002) [0.784..0.784] | 216.3 (0.14) | 416 / 0 | 5669 / 346 | 10 | 4961ecd4 = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 1 (108 tok) | 0.773 (0.0000) [0.773..0.773] | 216.2 (0.10) | 476/477 / 0 | 5669 / 346 | 10 | 1be9e33a = host (played run: same) | 1 |
| f8:5b:1b:91:6d:a8 | cea7be6+wt | row | female set 2 | firmware ramp 10,11,12,14,18,24 | demo 2 (175 tok) | 0.776 (0.0001) [0.776..0.777] | 216.3 (0.15) | 392 / 0 | 5669 / 346 | 10 | 98faff55 = host (played run: same) | 1 |

Calibration sentence (84 tokens, 5.59 s), 11 runs per cell, golden self-test PASS in every run: full split main 183.1 ms / RTF 0.662, int4 178.8 / 0.648, light 167.0 / 0.608; row split main 236.3 / 0.844, int4 231.3 / 0.826, light 216.3 / 0.776 (the first row-light cell has one outlier run during a split change, sd 17.8 ms). Hashes of all cells equal the host's (`ref/demo_ref`: female main a39e1a7b f6afba59 18154ad5; int4 0284b26f 251a24a1 d18ddec4; light 4961ecd4 1be9e33a 98faff55).

Memory over 60 minutes of runs: free PSRAM stayed at 3098 KB and free internal SRAM at 53 KB after every `rep`, `demo` and `test` (no leak); `rep` prints the free memory before and after (`REP memory` lines in the logs).


### Chunk size, clocks, kernels and the row to full split gain

(A) Chunk sweep on this board (female main set, calibration sentence 5.59 s, fixed chunk size, `first` = chunk, 5 runs per cell, sd of the RTF <= 0.0005):

| Steady chunk (frames) | chunks | weights read MB | MB per chunk | row RTF | row first chunk ms | full RTF | full first chunk ms |
|---|---|---|---|---|---|---|---|
| 8 | 56 | 151.1 | 2.70 | 1.032 | 218.5 | 0.821 | 168.0 |
| 12 | 38 | 102.2 | 2.69 | 0.917 | 252.1 | 0.715 | 195.4 |
| 14 | 32 | 102.6 | 3.21 | 0.919 | 268.0 | 0.722 | 208.4 |
| 16 | 28 | 90.4 | 3.23 | 0.892 | 286.1 | 0.699 | 223.5 |
| 18 | 25 | 81.6 | 3.26 | 0.873 | 309.5 | 0.691 | 243.8 |
| 20 | 23 | 74.9 | 3.26 | 0.857 | 333.6 | 0.678 | 261.9 |
| 24 | 19 | 64.3 | 3.38 | 0.833 | 377.3 | 0.655 | 296.6 |

The bytes per chunk jump by 19 % between 12 and 14 frames on this board too (2.69 to 3.21 MB). Cause: `WIDE_ROWS = 12` in `engine/itofs.c`: the two wide layers (harm_proj and the output head, 1202 columns) process 12 rows per weight pass, so a chunk of more than 12 frames reads their weights twice. It is neither the ramp (there is none in this sweep) nor a counter issue. A fit of the whole-sentence time gives about 13.2 ms per MB of weights read (about 76 MB/s effective, 86-89 MB/s measured peak) and no per-chunk fixed cost (-1 ms per chunk; the 12 to 14 frame step has 6 fewer chunks at the same bytes and the time does not fall). The first chunk costs 132 ms + 10.0 ms per frame (row) and 98 ms + 8.1 ms per frame (full).

(B) Per-chunk times, row split, main set, demo sentences (trace of this session, `s/nov02_final_boot5.log`): chunk times are 227 ms for a 24-frame chunk with a slow chunk of 340 ms (the text side advancing 24 tokens) at chunk indices 8, 13, 19, 22, 27, 31 and 34 depending on the sentence; the 25-token sentence has 10 chunks, the 108-token one 27, the 175-token one 38, and the plan (511 / 696 / 646 ms) is computed from the calibration trace by chunk index, so it follows where the slow chunks fall, not the length: the 108-token sentence meets a 340 ms chunk at index 8 while the buffer is thin, the 175-token one at index 8 too but with a different earlier chunk (the first ramp chunks differ by the text step). The model bound `RTF steady 0.76 x 1.25` applies to the steady chunks only; the plan also covers the ramp and the slow chunks. The dump is in the log; a cleaner per-chunk model is not done.

(C) CPU frequency and PSRAM clock, same build, main set, fixed 8-frame chunks, demo 1 (108 tokens), 5 runs:

| Config | one core RTF | row split RTF | full split RTF |
|---|---|---|---|
| 240 MHz, PSRAM 80 MHz | 1.284 | 1.034 | 0.823 |
| 160 MHz | 1.762 (+37.2 %) | 1.380 (+33.5 %) | 1.080 (+31.2 %) |
| PSRAM 40 MHz | 1.593 (+24.1 %) | 1.342 (+29.8 %) | 1.120 (+36.1 %) |

With the light set and the shipped ramp (demo 1): 240 MHz 1.019 / 0.775 / 0.609, 160 MHz 1.464 / 1.093 / 0.861, PSRAM 40 MHz 1.148 / 0.906 / 0.711.

(D) The cycle benchmark of `kernel_check_s3` (exactness: 972 of 972 shapes identical to the C reference on every variant) gives, in cycles per MAC with 8 rows: weights in SRAM 0.21-0.43, weights in PSRAM 0.33-0.59 (in=768: 0.209 against 0.527; in=256 at 20 rows: 0.243 against 0.371). The boot benchmark of the engine's own shapes (GMAC/s, 8 rows, weights in PSRAM, direct): one core 0.42-0.45; two cores 0.62-0.67 (full split); weights resident in SRAM, one core: 0.80. So one core is stalled for roughly 45 % of its GEMM time by PSRAM, and a second core adds 1.5x, not 2x. Hardware stall counters were not read (no counter access built).

(E) Row to full split: 0.845 to 0.661 is -22 % on the calibration sentence with the ramp; the GEMM alone goes from 0.42-0.45 to 0.62-0.67 GMAC/s (+48 %) on the same bus, consistent with a shared PSRAM bus, while the non-GEMM time (47 % of the row-split time) is unchanged. A -31 % prediction needed a 2x GEMM scaling. Whether the limit is the bus itself or the cache refill ordering was not separated.

(F) Core 1 in the row split: not measured by a counter. From the boot benchmark, GEMM calls take 53 % of the row-split wall time (944.7 of 1793.9 ms for demo 0; core 1 idle then) and the rest is split across both cores, so core 1 is busy at most 47 % of the time, "about a third" is consistent with part of that being single-core. `S3_PROF` was not built.

### Comparison with the numbers published above (his value, this board, same configuration?, possible reason)

| Item | Published value | This board | Same configuration? | Reason / note |
|---|---|---|---|---|
| Full split first chunk / RTF / start | 184 ms / 0.66 / 246 ms (fedbd26) | 183.1 ms / 0.661 / 246 ms (main 9c8105d, full split, IDF 5.5.1) | same board class, current main | reproduced |
| Row split RTF, 24 frames | 0.836 (second board) | 0.833, 64.3 MB read | same plan | reproduced |
| Row split RTF, 8 / 12 / 16 frames | 1.056 / 0.932 / 0.903 | 1.032 / 0.917 / 0.892 (runtime `chunk n`); 1.045 / 0.901 with compile-time 8 / 16 builds | same plan, other board | within 1-2 %; the run-time and compile-time 8-frame builds differ by 1.3 % |
| First chunk, row, 8 frames | 246 ms | 218.5 (run time) / 243.3 (compile time 8) | | the compile-time build reproduces; the run-time chunk keeps the 24-frame arena |
| Bytes per chunk jump 12 to 16 frames | +18 % (2.80 to 3.33 MB) | +19 % (2.69 to 3.21 MB at 14) | | `WIDE_ROWS = 12` |
| Per-chunk fixed cost | about 35 ms with 13 ms per MB | about 0 ms per chunk, 13.2 ms per MB | | not reproduced here; the data are collinear (chunks and bytes), the 12 to 14 step separates them |
| PSRAM bandwidth memcpy / GDMA / cache | 86 / 36 / 89 MB/s | 86.0 / 35.3 / 89.2 | | reproduced |
| GDMA slower than direct reads | RTF 1.250 against 1.056 at 8 frames | boot bench RTF direct 0.849, copy 0.870, gdma 0.948 (row, ramp) | | reproduced: direct wins |
| Internal SRAM peak | 354 of 374 KB (fedbd26) | 354 (main full split, IDF 5.5.1), 366 (prebuilt row, IDF 5.5.5), 346 of 371 (branch with ES8388) | different builds | layout differs by build; the margin of the prebuilt is 8 KB, so 354 does not describe the prebuilt |
| int4 / light, full split | RTF 0.65 / 0.61, first chunk 180 / 168 ms | 0.648 / 0.608, 179.3 / 167.7 ms | | reproduced |
| int4 / light, row split | not published | 0.826 / 0.776, first chunk 231.7 / 216.5 | | new |
| 160 MHz one core / row | +37 % / +46 % | +37.2 % / +33.5 % | main set, 8 frames | one core reproduced; the row figure differs (other board, other harness) |
| PSRAM 40 MHz one core / row | 1.642 / 1.711 (row slower than one core) | 1.593 / 1.342 (row faster) | main set, 8 frames | one core reproduced (+24 %); the row-slower-than-one-core result is not reproduced |
| Kernel check | all identical | 972 of 972 identical | | reproduced (cycles per MAC in 5(D)) |
| Stress test | 0 failures (one board, 21 min) | one round, 422 s, 6 configurations, 0 mismatches, 0 brownouts | same program | reproduced for one round |
| Planned start 513 / 697 / 657 | | 513 / 697 / 657 | same | reproduced; see 5(B) for the shape |

Not touched because nothing looked wrong: the second board and its supply history, the brownout ladder thresholds (need a scope), the instruction/data cache and line sweeps, the GDMA 8 KB tile, the 20-frame plan, the arena placement table; PSRAM at 120 MHz does not build with this module's QIO flash in ESP-IDF 5.5.1 (the timing tables need octal flash).

