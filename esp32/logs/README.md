# Raw serial logs from the physical board

All logs come from one ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), no DAC or amplifier attached, nothing listened to. They are the unedited console output; numbers in the READMEs are read from them. "First audio" in the READMEs is the first 125 ms audio chunk computed and ready after the phoneme ids are handed to the engine (text-to-phoneme conversion runs on the host and is not included), not audible latency.

No boot behind a published number is a cold power cycle: the reset reason in those logs is `USB_UART_CHIP_RESET` (a reset over the USB serial port or by the watchdog), not a power-on reset.

## 8 October 2026

| File | What it is |
|---|---|
| `board_crash_prebuilt_original_2026-10-08.log` | The first public prebuilt image on first boot: crash loop (`Guru Meditation Error`, LoadProhibited; GDMA self-test heap overflow). Kept as evidence of the bug; no numbers are taken from it. |
| `board_crash_gdma_selftest_2026-10-08.log` | A longer capture (28 crash-loop reboots) of the crashing build while the GDMA self-test was investigated. No numbers are taken from it. |
| `board_run1_bench_nomem_2026-10-08.log` | A partial capture (no boot header) of an early run on the board, with the female voice. Not used for the published numbers. |
| `board_fixed_boot_2026-10-08.log` | The fixed image (commit `fedbd26`), female voice, one boot: self-test PASS, calibration, three demo sentences. |
| `board_fixed_boot_and_tiers_2026-10-08.log` | The same boot, continued: the female voice's other two weight sets timed by hand once each with `tier`. This is the source of the 8 October numbers. |

## 9 October 2026 (same board, same firmware `fedbd26`)

| File | What it is |
|---|---|
| `male_boot_2026-10-09.log` | Male voice, one boot. The boot calibration chose set 0 (main int8, RTF 0.659); self-test PASS (137,400 samples, fnv32 `35f96c0b`, identical to the host C engine). |
| `male_tiers_2026-10-09.log` | Male voice, a second boot (same result: set 0, RTF 0.659), then the other sets timed by hand with `tier`: set 1 (main int4) in two passes, set 2 (light) in one pass, and set 0 again. Hand-timed sets are not boot-chosen and the firmware plans no start delay for them (`planned -1`). The serial echo of typed commands is interleaved with the output of the previous command, so read each `TIMING` line's `weight set` field rather than the `>>>` echoes. |
| `male_tiers_ABORTED_overlapping_cmds_2026-10-09.log` | An earlier attempt to time the male voice's sets. Aborted because commands overlapped each other and the reboot after the port was opened. **Not used for any number.** |
| `male_tiers_ABORTED_cmds_sent_during_reboot_2026-10-09.log` | Another aborted attempt: commands were sent during the reboot that follows opening the port. **Not used for any number.** |
| `female_boot_wdt1_2026-10-09.log` | Female voice, repeat boot 1 of the identical image (reset, not a cold power cycle). |
| `female_boot_wdt2_2026-10-09.log` | Female voice, repeat boot 2. |
| `female_stats_2026-10-09.log` | Female voice, repeat boot 3, with extra `demo` runs and the `stats` command (PSRAM peak 5372 of 8192 KB, internal SRAM 354 of 374 KB). Stack high-water marks were not printed in this log. |

Over the four female boots (8 October and these three) the self-test passed every time (fnv32 `023bd6b1`, identical to the host), and underruns were 0 in every run in every log above that is used for numbers.

## 9 October 2026, stability runs (same board, firmware `fedbd26` for the soak)

| File | What it is |
|---|---|
| `soak_female_240MHz_2026-10-09_0343.raw.log` | Raw serial capture of the soak (female voice, 240 MHz, USB powered, nothing attached, I2S idle), about 2.5 hours. |
| `soak_female_240MHz_2026-10-09_0343.events.log` | The soak script's own progress log (timestamps, per-10-iteration counts, phase summaries). Its one "CRASH/REBOOT MARKER" line is the boot banner that appears when the port is opened (the board resets then), at 03:43:14 before the first phase; there is none later. |
| `soak_female_240MHz_2026-10-09_0343.summary.txt` | The four phase summaries: main int8 set 300 iterations (900 PASS, 0 FAIL, hashes `023bd6b1` and `9666bb9e`), int4 set 100 iterations (300 / 0, `363d7d98`), light set 100 (300 / 0, `5f12d5e1`), main int8 again with I2S idle 100 (300 / 0). Each iteration runs the golden sentence in three modes (dual-core 8-bit, single-core, dual-core 16-bit) and compares the hash with the host C engine. The `supply_idle` phase name means I2S idle; the USB voltage was not measured. |
| `ABORTED_prelim_soak_*_2026-10-09_0313.log` | A first soak attempt that was stopped; **not used for any number**. |
| `synthetic_stress_240MHz_2026-10-09.log` | The synthetic two-core stress (source in `esp32/tools/stress/`): three full rounds (about 21 minutes) and the start of a fourth, 0 mismatches. The capture was stopped part-way through the fourth round. |
| `dualcore_load_comparison.md` | A reading of the code (not a measurement) that compares how Ito and Oído load the two cores. |

## Other logs (not from the board)

`qemu_*.log` in a local checkout are QEMU emulator runs; they are ignored by git and not part of this repository.
