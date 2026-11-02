# Two-core PIE load and the 3.3 V supply (ESP32-S3)

**Short version.** Ito runs its int8 GEMM in one of two ways: the **row split** (safe, slower) and the **full split** (both cores in the GEMM, about 28 % faster, and the figures in the root README). Running dense PIE (`EE.*`) integer GEMM on both cores of an ESP32-S3 at 240 MHz can draw enough current to pull the 3.3 V rail down on some development boards: weak regulators, long or thin USB cables, or too little bulk capacitance next to the module. The result is not a clean crash; it looks like a software bug. **The firmware now decides at run time, per board** (see "Adaptive mode" below): it starts in the row split, runs a short supply self-test on the real kernels, and uses the full split only if the test passes and nothing has gone wrong since. Any brownout, watchdog or panic reset, or hash mismatch while the full split is active latches the board back to the row split until the owner says otherwise. The compile-time option is now only the initial policy. Output audio is the same, bit for bit, in every mode.

## Adaptive mode (default)

Three parts: a verdict, a guard against hangs, and console control.

**The verdict** lives in one NVS record (namespace `ito`, key `split`, a versioned and checksummed blob, `engine/itofs_split.h`). It holds the owner's preference (`auto`, `row`, `full`), the supply verdict (`untested`, `pass`, `fail`), the reason code of the last change, and counters (boots, tests, passes, failures, brownout resets, watchdog or panic resets, hash mismatches, interrupted tests).

| Situation | What the firmware does |
|---|---|
| First boot, `auto`, no verdict | row split while the **long supply test** runs (about 3 s: the full split on the engine's own GEMM shapes, weights streamed from PSRAM, every result compared bit for bit with the one-core result of the same call). The record is written with `testing = 1` **before** the test. A pass stores `pass` and engages the full split; the golden-sentence self-test (`fnv32 023bd6b1`) then runs in the full split. |
| Reset during the long test with a brownout, watchdog or panic reason | the next boot reads `testing = 1` and the reset reason: verdict `fail` (reason `supply test: hang / brownout / panic`), row split. |
| Power cut or ordinary reset during the test | no verdict (counted as `interrupted`), the test runs again. |
| Later boots, verdict `pass` | **quick re-check** (about 0.25 s, same test, fewer passes) at every boot. A mismatch sets `fail` and the row split. |
| A brownout, watchdog or panic reset while the full split was engaged | verdict `fail`, reason `brownout / watchdog / panic reset in full split`, row split from then on (the record says the full split was active when the board went down). |
| Hash mismatch of a self-test in the full split | verdict `fail`, row split, the same self-test is repeated in the row split. |
| Reset of any of those kinds in the row split | counted, no verdict. |
| `fail` | stays until the owner sends `split auto`, `split row` or `split full`, or the NVS record is erased (`esptool erase-region 0x9000 0x6000`). |

`split` prints the mode, the preference, the verdict and why, and the counters. `split auto` clears a latch and runs the long test now; `split row` forces the row split; `split full` forces the full split without a test (the guard still applies: a brownout or hang while forced latches the row split, so a failing supply cannot cause a boot loop). Each of them re-measures the weight sets in the new split (`recal`). `reboot` restarts the board with a software reset, which is not held against the full split (an esptool watchdog reset after flashing is; erase the NVS record, or run `split auto`, after flashing over a board whose full split is engaged).

**Hang recovery.** The task watchdog stays off (a 0.25 s chunk would starve the idle task by design). Instead the RTC watchdog is armed at 15 s, fed once per synthesised chunk, in every benchmark loop and once a second while the console is idle; during the supply test it is tightened to 4 s. A hang with interrupts on or off, in code or in a PIE instruction, resets the whole chip; the reset reason (`watchdog`) goes into the record as above. The brownout detector stays enabled (level 7) and its resets are counted. A test build (`-D ITOFS_EXTRA_DEFS="ITOFS_INJECT_HANG=1"`) has a `hang` and a `hang irq` console command to check it.

**Initial policy.** menuconfig `Ito` then `Initial policy of the GEMM split`: `auto` (default), `row` (never tests; `split full` can still force it) or `full` (no test; the guards still apply). `-D ITOFS_EXTRA_DEFS="S3_GEMM_DUAL_DEFAULT=0"` or `=1` forces row or full. The policy applies when the record is new or was made by a build with another policy.

QEMU has no PIE current draw, so the supply test is a no-op PASS there (the log says so) and the dual-core path is exercised by the self-test instead.

## The two modes

Both modes run the same engine on the same weights and produce the same PCM. They differ only in which core executes the int8 GEMM calls (the matrix multiplies that hold nearly all the multiply-accumulates).

| Work | Row split (`s3_gemm_dual = 0`, safe, the start value) | Full split (`s3_gemm_dual = 1`, after a passed supply test) |
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

- A silent hang at the first synthesis call, with no panic. The task watchdog is off by design (long synthesis loops); the firmware's RTC watchdog now resets the board after 15 s (4 s during the supply test) and the reset is recorded (see "Adaptive mode"). On a firmware without it, a hang stays a hang until you reset the board.
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

1. **Leave the default (`auto`).** The firmware tests your board and keeps the row split if the test fails. Read `split` after the first boot: `mode=full verdict=pass` means this board's supply passed.
2. **If you want to be sure, run the stress test** (see "Check your board") before relying on the full split in the field; the firmware's own test is short (3 s) and is a screen, not a proof.
3. **Improve the supply** if you want the full split on a board that fails:
   - 220 to 470 uF of bulk capacitance on 3V3 as close to the module as you can (low-ESR electrolytic or polymer tantalum, plus the usual ceramic);
   - a 3.3 V regulator rated for at least 1 A (many development boards use a small regulator that is marginal at this load);
   - a short, thick USB cable, a different host port or a powered hub, or a 5 V feed on the 5 V pin from a supply that can deliver an ampere;
   - or a different board.
   Changing the cable alone did not fix the board that failed. After a hardware fix, `split auto` runs the test again.
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

## Choosing the split by hand

- At run time: `split auto|row|full` (stored in NVS, survives resets). `split` shows the state.
- At build time, as the initial policy only: menuconfig `Ito` then `Initial policy of the GEMM split`, or `CONFIG_ITOFS_SPLIT_ROW=y` / `CONFIG_ITOFS_SPLIT_FULL=y` in `sdkconfig.defaults`, or `-D ITOFS_EXTRA_DEFS="S3_GEMM_DUAL_DEFAULT=0"` / `=1` (the definition wins over menuconfig). Rebuild from a fresh build directory so that CMake and Kconfig pick the change up. (`CONFIG_ITOFS_GEMM_FULL_SPLIT` of earlier builds no longer exists; it is ignored.)

The boot log prints the last reset reason, the `SPLIT mode=... verdict=...` line, and the `SUPPLY_TEST` result. The boot self-test, the board benchmark, the boot calibration and every synthesis follow the active mode, and `TIMING` lines name it (`split row` / `split full`).

## Measured on the first board, 2 November 2026

With the adaptive firmware on the board of the root README (DevKit N16R8 on an audio shield, speaker playing): the 3 s test passed (273 passes of the full split, all equal to the one-core result), the quick re-check passed at every boot (23 passes in 270 ms), one round of the stress program (422 s, six configurations) had 0 mismatches, and the full split ran for hours of synthesis (about 400 utterances with their PCM hashes equal to the host's) with 0 brownout and 0 watchdog resets, also with the codec at +4.5 dB and the speaker playing. A hang injected in the full split was reset by the RTC watchdog after 13.9 s and the next boot ran the row split with `why="watchdog reset in full split"`; with interrupts off the interrupt watchdog reset it after 0.4 s. The row split on this board: RTF 0.843 against 0.661 for the full split with the main set, first chunk 237 against 183 ms, planned start 511-697 against 246 ms. Details: [`BENCHMARKING.md`](BENCHMARKING.md), last section.

## Limits of this finding

- One failing board, one session of stress tests per case, a handful of retests. The threshold ladder has seven coarse steps.
- Whether a given board fails depends on its regulator, its bulk capacitance, its cable and its host port. The published full-split Ito numbers were taken on a board where the full split is stable (see the root README, "Stability on the board"). Both statements are true.
- Nothing here was measured with a scope or a current probe.
