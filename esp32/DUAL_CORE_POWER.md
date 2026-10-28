# Two-core PIE load and the 3.3 V supply (ESP32-S3)

**Short version.** Ito runs its int8 GEMM in one of two ways, and **the default is the safer, slower one: the row split**. The faster **full split** is an opt-in. Running dense PIE (`EE.*`) integer GEMM on both cores of an ESP32-S3 at 240 MHz can draw enough current to pull the 3.3 V rail down on some development boards: weak regulators, long or thin USB cables, or too little bulk capacitance next to the module. The result is not a clean crash. It looks like a software bug. If your board hangs, resets or gives wrong results only when both cores run the GEMM, you are on the wrong side of this page. The default build reduces the load, because only one core runs the dense GEMM at a time. It does not fix a failing supply: on the affected board the row split ran for weeks, but later even single-core int16 runs stopped once the supply degraded further.

**Breaking change.** Earlier builds, including the earlier prebuilt image, defaulted to the full split, and the headline numbers in the root README were measured with it. The default is now the row split. Speed on a board with a good supply drops (RTF about 0.66 with the full split and the shipped chunk ramp, about 0.84 to 0.90 with the row split on another board and fixed chunk sizes; see "Cost" below for exactly what each number is). Output audio is unchanged, bit for bit. To get the old behaviour back, see "Opting in to the full split".

## The two modes

Both modes run the same engine on the same weights and produce the same PCM. They differ only in which core executes the int8 GEMM calls (the matrix multiplies that hold nearly all the multiply-accumulates).

| Work | Row split (default, `s3_gemm_dual = 0`) | Full split (opt-in, `s3_gemm_dual = 1`) |
|---|---|---|
| int8 GEMM (PIE dot products) | core 0 only | core 0 and core 1, each half of the output channels |
| Row-parallel float loops (GELU, normalisation, depthwise convolutions, source features) | core 0 and core 1, rows split in half | core 0 and core 1, rows split in half |
| Everything else (scheduling, text side, I2S) | core 0 | core 0 |
| Dense SIMD load on the 3.3 V rail | one core at a time | both cores at the same time |
| Speed | slower (core 1 is idle for the GEMM, about two thirds of the time) | faster |
| Needs a stiff 3.3 V supply | no more than one core of PIE | yes, tested on your board |

```
row split          core 0: [ GEMM (all channels) ][ half the rows ][ GEMM ][ half the rows ] ...
                   core 1: [      idle           ][ half the rows ][ idle ][ half the rows ] ...

full split         core 0: [ GEMM (half channels) ][ half the rows ][ GEMM (half) ][ half the rows ] ...
                   core 1: [ GEMM (half channels) ][ half the rows ][ GEMM (half) ][ half the rows ] ...
                             ^ both cores in dense PIE at once: the peak current
```

Why bit-identical: the GEMM accumulates in exact int32 arithmetic, and each output channel is computed by exactly one core in either mode (the weight stream follows the split). The float loops are the same in both modes. So there is no reordering of any floating-point operation. This was checked, not assumed: the firmware's self-test hashes the PCM of a golden sentence (84 tokens, 134,100 samples) and compares it with the host C engine, and it prints `fnv32 023bd6b1` in both modes (and `9666bb9e` with 16-bit activations); a full sample-by-sample comparison of the dumped PCM against the host engine in QEMU gives 0 differing samples in both modes. The boot log states which mode is active (`GEMM split: row ...` or `GEMM split: full ...`).

## Cost

Numbers are real-time factor (compute time over audio duration; below 1 is faster than real time). They come from **different boards and different harnesses** and are not one controlled comparison, so read each with its chunk plan.

| Mode | RTF | Chunk plan | Board and harness |
|---|---|---|---|
| Full split | about 0.66 (0.660 to 0.674 on the demo sentences) | the firmware's ramp: a 125 ms first chunk, then 11, 12, 14, 18 and 24 frames | a board with a good supply, the shipped firmware's own boot calibration |
| Row split | about 0.90 (0.903) | 16 frames per chunk, fixed | a second board of the same module class, a separate harness driving Ito's kernels |
| Row split | about 0.84 (0.836) | 24 frames per chunk, fixed | the same second board and harness |
| One core, no split | about 1.32 (1.323) | 8 frames per chunk, fixed (the row split is 1.056 at the same 8 frames) | the same second board and harness |

The full-split and row-split numbers are not a like-for-like pair: the ramp reaches 24-frame chunks only at the end, and the firmware's own row-split RTF on the first board was not measured. Expect the row split to land somewhere from about 0.84 to 0.9 with 24 to 16-frame chunks; a board-specific figure comes from running the boot benchmark yourself. More chunk plans, and why the plan matters, are in [`BENCHMARKING.md`](BENCHMARKING.md). The firmware's boot calibration (`CALIB`, `TIER_SELECT`) measures whichever mode is active, so it picks a lighter weight set automatically if the row split is too slow for real time on your board.

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
| an errata item | the [published ESP32-S3 SoC errata](https://docs.espressif.com/projects/esp-chip-errata/en/latest/esp32s3/03-errata-description/index.html) contain no matching item: none covers PIE, PSRAM, MSPI, temperature or a dual-core hang. The only cache item, CACHE-126 (a hit error during a software-triggered write-back), is unrelated and is worked around by ESP-IDF 5.5 (`esp_rom/patches/esp_rom_cache_writeback_esp32s3.S`). The dependence on load density and frequency points at the supply |

Two further notes. One core of dense PIE on internal SRAM already browns out on the failing board, so the problem is not specific to two-core code. And the lazy save of the coprocessor state in ESP-IDF was checked both in the source (ESP-IDF v5.5.5) and by experiment (the two-task test above):

- `tie.h` for the ESP32-S3 defines `XCHAL_CP_MASK` as `0x09`: coprocessor 0 (the FPU) and coprocessor 3 (the AI/PIE coprocessor), whose save area is 208 bytes, 16-byte aligned.
- `_xt_coproc_exc` in `components/xtensa/xtensa_vectors.S` saves the state lazily, on first use after a switch, with a per-core owner array guarded by a cross-core spinlock.
- `_frxt_coproc_exc_hook` in the FreeRTOS `portasm.S` pins a task to its core on its first coprocessor use.
- `vPortCleanUpCoprocArea` in the FreeRTOS xtensa `port.c` releases the ownership when a task is deleted.

## What to do

1. **Leave the default (row split) unless you have tested your board.** Nothing to configure. The GEMM stays on core 0 and only the row loops use both cores.
2. **Test your board, then opt in if it passes** (see "Check your board" and "Opting in to the full split").
3. **Improve the supply** if you want the full split on a board that fails:
   - 220 to 470 uF of bulk capacitance on 3V3 as close to the module as you can (low-ESR electrolytic or polymer tantalum, plus the usual ceramic);
   - a 3.3 V regulator rated for at least 1 A (many development boards use a small regulator that is marginal at this load);
   - a short, thick USB cable, a different host port or a powered hub, or a 5 V feed on the 5 V pin from a supply that can deliver an ampere;
   - or a different board.
   Changing the cable alone did not fix the board that failed.
4. **160 MHz is a software remedy**, safe on the failing board but slower than the row split at 240 MHz (at 8-frame chunks on the second board: RTF 1.547 for the row split at 160 MHz against 1.056 for the row split at 240 MHz; both are the row split). The full split at 160 MHz on that board measured RTF 1.121 at chunk 8. It passed in those runs, but in other reruns it hung twice, so 160 MHz lowers the load without removing the failure. It is not recommended unless you need the power savings anyway.

## Check your board

Run the stress test before enabling the full split. [`tools/stress/`](tools/stress/README.md) is a small ESP-IDF program that runs Ito's PIE kernel on both cores and compares every result with a one-core reference.

```bash
cd esp32/tools/stress
idf.py set-target esp32s3
idf.py build flash monitor      # leave it running for several minutes
```

It starts the second core with a spin flag and uses random data, so it exercises the supply but is gentler than a sustained dense load over a long run. For a board you intend to ship, use something at least as demanding as these cases at 240 MHz:

1. Both cores running the dot rows over a buffer in PSRAM, 10 seconds, every pass checked.
2. The same with the buffer in internal SRAM (no stalls: the worst case).
3. The same cases at 160 MHz as a control.
4. Optionally, the brownout-detector ladder above, to see how far the rail falls.

Pass means no reset, no hang, no `Brownout detector was triggered` message and no mismatch in any pass. If the PSRAM case on both cores hangs or mismatches, keep the row split. Test with the board powered the way it will be used (the same cable, port or supply, and with the radio, display or I2S load you will have), because the margin moves with it.

## Opting in to the full split

Two equivalent ways, both at build time. Rebuild from a fresh build directory so that CMake and Kconfig pick the change up.

1. **menuconfig (preferred).** `idf.py menuconfig`, then `Ito` then `Run the int8 GEMM on both cores (full split)`. Or add `CONFIG_ITOFS_GEMM_FULL_SPLIT=y` to your `sdkconfig.defaults`. Rebuild with `esp32/tools/build_fw.sh`, or by hand in `esp32/firmware`.
2. **A compile definition.** `idf.py -B build -D ITOFS_EXTRA_DEFS="S3_GEMM_DUAL_DEFAULT=1" build`. The definition wins over the menuconfig option, in both directions (`S3_GEMM_DUAL_DEFAULT=0` forces the row split).

The boot log prints `GEMM split: full (...)` or `GEMM split: row (...)`, so you can see which one a running image uses. The boot self-test, the board benchmark, the boot calibration and every synthesis all follow the active mode. The prebuilt image is built with the row split; the full split needs a rebuild.

## Limits of this finding

- One failing board, one session of stress tests per case, a handful of retests. The threshold ladder has seven coarse steps.
- Whether a given board fails depends on its regulator, its bulk capacitance, its cable and its host port. The published full-split Ito numbers were taken on a board where the full split is stable (see the root README, "Stability on the board"). Both statements are true.
- Nothing here was measured with a scope or a current probe.
