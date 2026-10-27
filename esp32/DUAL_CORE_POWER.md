# Two-core PIE load and the 3.3 V supply (ESP32-S3)

**Short version.** Running dense PIE (`EE.*`) integer GEMM on both cores of an ESP32-S3 at 240 MHz can draw enough current to pull the 3.3 V rail down on some development boards: weak regulators, long or thin USB cables, or too little bulk capacitance next to the module. The result is not a clean crash. It looks like a software bug. If your board hangs, resets or gives wrong results only when both cores run the GEMM, read this page before debugging the engine.

Ito's two-core GEMM (`s3_gemm_dual = 1`, the default) is exactly this load. It runs fine on the board used for the published numbers (see the root README), and it hangs on at least one other ESP32-S3-WROOM-1 N16R8-class development board. The safe alternative, the row split, costs speed (below) and is bit-identical.

This page is written from stress tests on one board that failed, with plain ESP-IDF code and no Ito engine involved. It is a measurement on one board, not a statement about any particular board design.

## Symptoms

Any of these, often differing from run to run and moving when you change code layout, memory placement or timing:

- A silent hang at the first synthesis call, with no panic. The task watchdog does not recover it (Ito's `sdkconfig.defaults` turns the task watchdog off for the long synthesis loops, so a hang stays a hang until you reset the board).
- `E BOD: Brownout detector was triggered` on the console, then a reset, again after every reset.
- An occasional `Cache error: MMU entry fault`.
- Wrong results: the output hash differs from the host engine, sometimes only on some passes.
- A run that passes with one core, at 160 MHz, with a different build or with the weights in another memory, and fails when you change something unrelated.

Because the failure follows power and not code, "it passed on the previous binary" proves little: a different cable, USB port or load on the host can change it.

## What was measured

A small ESP-IDF program with no model in it. Both cores stream a buffer for 5 to 10 seconds per case, each doing either plain `l32i` loads or the GEMM's dot rows (`ee.vld.128` plus `ee.vmulas.s8.accx`), over PSRAM, over the flash-mapped app image, or over internal SRAM. Every pass is checked against a one-core reference.

### Results at 240 MHz

| Case | Result |
|---|---|
| one core, loads, PSRAM or SRAM | pass |
| two cores, loads, PSRAM | pass |
| two cores, loads, internal SRAM | borderline: passed in one session, browned out in a retest |
| one core, PIE dot, PSRAM or flash | pass |
| two cores, PIE dot, flash (about 20 ms per 512 KB pass) | pass |
| one core, PIE dot over 32 KB in internal SRAM (no memory stalls, the densest load) | brownout reset within the first second, every time |
| two cores, PIE dot, PSRAM (the shape of the GEMM) | silent hang, not recovered by a watchdog |
| two cores, PIE dot, internal SRAM | brownout reset |
| two cores, PIE dot, PSRAM, about 50 % idle time (a busy-wait after every 1 KB row) | no hang, but wrong sums on a few percent of passes |
| the same with about 34 % idle | hang |
| the same with 60 % or more idle | pass |

At 160 MHz every case passes except two cores of PIE over internal SRAM, which still resets the board. The failure follows the power drawn (activity times frequency), not the memory the data lives in.

### Brownout-detector ladder

The ESP32-S3 brownout detector has seven thresholds (ESP-IDF documents them as estimates between about 2.44 V and 3.30 V). Arm each one in turn with the reset disabled and read the latched interrupt flag after each load phase:

- Idle: the supply stays above 3.19 V.
- One core of PIE over PSRAM, and two cores of PIE over flash: cross 3.19 V, do not cross 2.98 V.
- The failing loads (two cores of PIE over PSRAM or SRAM, one core over SRAM): go below 2.44 V, the lowest threshold.

A step from about 0.2 V of droop to more than 0.8 V between two loads that differ by one core looks like a current limit being reached, not like a resistive drop. These are threshold crossings of an on-chip detector, not oscilloscope measurements: put a scope on 3V3 at the module if you need real numbers.

## Why it is not the software

| Hypothesis | Evidence |
|---|---|
| a bug in the engine or its kernels | a stress program with none of the engine in it fails the same way |
| a cache, MMU or PSRAM fault | plain loads on both cores over PSRAM and flash are exact; the same PIE kernel over flash on both cores passes |
| PIE (coprocessor) state saved wrongly when tasks switch | the stress program runs two busy tasks, one pinned to each core, and nothing else, and still fails; plain loads on both cores pass. A context-switch bug would not depend on the load density |
| the task stack or the weight-staging mode | a larger worker stack does not change it; direct reads and GDMA staging both fail (GDMA shows the brownout message) |
| not enough core voltage | raising the digital core voltage within ESP-IDF's cap makes it worse (a borderline case goes from sporadic wrong sums to a hang): more voltage draws more current |
| temperature | the die sensor read 26 to 40 degrees C at the start and end of every case |
| an errata item | none of the published ESP32-S3 errata matches this pattern, and the dependence on load density and frequency points at the supply. This is absence of evidence, not a statement from the vendor |

Two further notes. One core of dense PIE on internal SRAM already browns out on the failing board, so the problem is not specific to two-core code. And the lazy save of the coprocessor state in ESP-IDF was checked by experiment (the two-task test above), not by auditing its source.

## What to do

1. **Use the row split as the default.** Keep the GEMM calls on core 0 and split only the row-parallel loops (GELU, normalisation, depthwise convolutions, source features) across both cores. This is `s3_gemm_dual = 0`, set at build time with `S3_GEMM_DUAL_DEFAULT=0` (see "Choosing the split" below). The output is bit-identical to the full split. On a second board with the weights in PSRAM and the engine's chunk sizes it ran at about RTF 0.90 at 16 frames per chunk (chunk size matters a lot, see [`BENCHMARKING.md`](BENCHMARKING.md)), against about 1.32 on one core.
2. **Use the full split only on a board whose supply you have tested**, and only after the stress test below passes for that board.
3. **Improve the supply** if you want the full split on a board that fails:
   - 220 to 470 uF of bulk capacitance on 3V3 as close to the module as you can (low-ESR electrolytic or polymer tantalum, plus the usual ceramic), or a stronger 3.3 V regulator;
   - a short, thick USB cable, a different host port or a powered hub, or a 5 V feed on the 5 V pin from a supply that can deliver an ampere.
   Changing the cable alone did not fix the board that failed.
4. **Run a stress test before you enable the full split** (below).
5. **160 MHz is a software remedy**, safe on the failing board but slower than the row split at 240 MHz (about RTF 1.12 against 0.90 on the second board), so it is not recommended unless you need the power savings anyway.

## Stress test

[`tools/stress/`](tools/stress/README.md) is a small ESP-IDF program that runs Ito's PIE kernel on both cores and compares every result with a one-core reference. It starts the second core with a spin flag and uses random data, so it exercises the supply but is gentler than a sustained dense load over a long run.

For a board you intend to ship, use something at least as demanding as these cases at 240 MHz:

1. Both cores running the dot rows over a buffer in PSRAM, 10 seconds, every pass checked.
2. The same with the buffer in internal SRAM (no stalls: the worst case).
3. The same cases at 160 MHz as a control.
4. Optionally, the brownout-detector ladder above, to see how far the rail falls.

Pass means no reset, no hang and no mismatch in any pass. If the PSRAM case on both cores hangs or mismatches, keep the row split.

## Choosing the split

`s3_gemm_dual` (declared in `firmware/main/kernels_s3.h`) is a runtime variable that defaults to 1. To make the row split the default of a build, define `S3_GEMM_DUAL_DEFAULT=0` through the firmware's `ITOFS_EXTRA_DEFS` hook: `idf.py -D ITOFS_EXTRA_DEFS="S3_GEMM_DUAL_DEFAULT=0" build` (a fresh build directory, so that CMake picks it up). The boot self-test, the benchmark and every synthesis then run the GEMM on core 0 and the row loops on both cores. The prebuilt image keeps the full split.

## Limits of this finding

- One failing board, one session of stress tests per case, a handful of retests. The threshold ladder has seven coarse steps.
- Whether a given board fails depends on its regulator, its bulk capacitance, its cable and its host port. The published Ito numbers were taken on a board where the full split is stable (see the root README, "Stability on the board"). Both statements are true.
- Nothing here was measured with a scope or a current probe.
