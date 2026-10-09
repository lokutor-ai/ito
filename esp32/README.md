# Ito on the ESP32-S3

The whole text-to-speech chain runs on an **ESP32-S3-DevKitC-1 N16R8** (16 MB flash, 8 MB octal PSRAM): text
encoder, duration head, prosody net, mel head, harmonic F0 source, ConvNeXt vocoder and iSTFT. Phonemisation runs on
the host that sends the text (`tools/say.py`). The board receives token ids and plays 24 kHz audio over I2S.

> **First run on a physical board (8 October 2026):** the first public prebuilt image crash-looped on first boot on real hardware: its GDMA self-test overflowed a heap buffer, which QEMU cannot show. That run found four first-silicon bugs (the GDMA self-test heap overflow, bench scratch fragmentation, underrun accounting, USB console input), fixed in `fedbd26` (with the notes in `e196730`). The prebuilt image in this repository is the fixed one, and the logs of the crashing runs are kept in `esp32/logs/board_crash_*.log`. The measured numbers below come from the fixed image.
>
> **Status (9 October 2026).** The engine and firmware are verified on the host and in Espressif's QEMU: the firmware's PCM is bit-identical to the host build, for all three weight sets of both voices. **Measured on a board** (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice): with the main int8 set the first audio chunk (125 ms of audio) was computed and ready 184 ms after the text was handed to the engine (183.9–184.3 ms over the three demo sentences), and the real-time factor was 0.66 on the boot calibration sentence (0.660–0.674 on the demo sentences). Gap-free playback is a separate number: the firmware plans to start playback about 246 ms after the text arrives, and with that planned delay the three demo sentences had 0 underruns. No DAC or amplifier was attached and no audio was listened to, so the I2S/DAC output stage is not included, these are not acoustic measurements, and the underrun count is the firmware's own accounting against its playout clock. The first audio chunk is ready in under 200 ms and synthesis runs faster than real time on this board (RTF 0.66), with 0 underruns when playback starts after the planned 246 ms delay. For the female voice, the other two weight sets were timed by hand once on 8 October with the `tier` serial command (one run each, not chosen by the boot calibration): main int4 set RTF about 0.65, first chunk about 180 ms; light set RTF about 0.61, first chunk about 168 ms. **Update, 9 October 2026** (same board, same firmware fedbd26, still no DAC and nothing listened to). Male voice: the boot calibration chose the main int8 set (RTF 0.659 on the calibration sentence, in both of two boots), the bit-exact self-test passed (137,400 samples, identical to the host C engine), the first audio chunk was ready in 183.8–184.8 ms over the demo sentences, the demo-sentence RTF was 0.659–0.672, the firmware planned a gap-free start 248–249 ms after the text arrives, and underruns were 0. The male voice's other two sets were timed by hand with `tier` (not boot-chosen; the firmware plans no start delay for hand-timed sets), demo sentences only: main int4 set first chunk 179.5–179.8 ms, RTF 0.645–0.653 (two passes); light set first chunk 167.9–168.0 ms, RTF 0.606–0.613 (one pass); the self-test passed with each set. Female voice, repeats: the same image booted four times in all (8 October and three times on 9 October, each after a USB-serial or watchdog reset, not a cold power cycle); the first chunk was ready in 183.5–184.5 ms over all demo runs, the boot-calibration RTF was 0.660–0.661, the planned start was 246–248 ms, the bit-exact self-test passed every time and underruns were 0. After a demo the PSRAM peak was 5372 KB (female) and 5380 KB (male) of 8192 KB, and internal SRAM 354 of 374 KB (stack high-water marks were not printed). All of this comes from one board: the repeats and the male runs show repeatability on one chip, not a second board. Against the pre-board estimates (optimistic / central / pessimistic: RTF 0.43–0.47 / 0.63–0.66 / 0.97–0.99, first audio chunk 124–132 / 171–180 / 251–260 ms), the measured RTF of 0.66 is at the top of the central estimate and far from the pessimistic one (0.97–0.99), and the measured first chunk of 184 ms is slightly above the central estimate and below the pessimistic one. Still not measured: audio through a DAC and any listening to the output (no DAC was attached), a second board or a second chip revision, cold power-on boots (every boot was a reset), power draw, long-run thermal behaviour and supply sensitivity, stack high-water marks (not printed), the male voice's sound quality through the chip (its audio is bit-identical to the host engine: self-test PASS), and a planned start delay for the hand-timed sets (the firmware plans none). Still estimates: the optimistic, central and pessimistic columns, which are the pre-board predictions kept for comparison. Details in §4a. All other timings in §3 and §4 are **pre-board estimates** from exact QEMU instruction counts and an assumed CPI and PSRAM bandwidth.
>
> **What the firmware does about it:** it carries up to three weight sets per voice (main int8, main int4, a light set with one block fewer), measures the real chunk times of each at boot (`CALIB`, `TIER_SELECT`), keeps the first set whose measured RTF is at most 0.85, plans the playback start delay from those measurements, and says so (`WARNING: DEGRADED`) if even the fastest set measures 0.95 or more (§4c). That is a check by measurement on the board that runs it, not a promise about other boards.

## 1. Flash and talk

Wiring for an I2S DAC or amplifier:
- BCLK to GPIO15, LRCK to GPIO16, DIN to GPIO17
- PCM5102A: SCK to GND and XSMT high
- or a MAX98357A on 5 V

```bash
pip install esptool pyserial phonemizer espeakng-loader nltk huggingface_hub
huggingface-cli login                                  # once, after accepting the terms at huggingface.co/lokutor-ai/ito
esp32/tools/fetch_weights.sh                           # -> models/ito_female_esp32s3{,_int4,_light}.bin (flash.sh runs it if needed)
esp32/tools/flash.sh /dev/ttyUSB0                     # prebuilt app at 0x0 + the voice's three weight sets at 0x200000 / 0x680000 / 0xB00000 (female voice)
VOICE=male esp32/tools/flash.sh /dev/ttyUSB0           # or the male voice
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # press RST and wait for READY (the boot benchmark takes 1-2 min)
python esp32/tools/say.py "Good morning! The coffee is ready." --port /dev/ttyUSB0
```

If flashing fails at 921600 baud, use `BAUD=460800`. If the board does not connect, hold BOOT, tap RST, release BOOT
and try again.

| file | what |
|---|---|
| `prebuilt/ito_app_merged.bin` (444 KB) | bootloader + partition table + app (ESP-IDF 5.5.1, octal PSRAM, QIO flash, I2S on). Flash at 0x0. |
| `ito_female_esp32s3.bin` (3.81 MB, from [Hugging Face](https://huggingface.co/lokutor-ai/ito), see [`models/README.md`](../models/README.md)) | female voice, **main set** (int8), with a self-test record and three demo sentences. Flash at 0x200000. **CC BY-NC-SA 4.0 + [`models/TERMS.md`](../models/TERMS.md): non-commercial.** |
| `ito_female_esp32s3_int4.bin` (3.20 MB, same place and license) | the same voice with int4 weights in the ConvNeXt blocks (§3b). Flash at 0x680000 (optional). |
| `ito_female_esp32s3_light.bin` (3.05 MB, same place and license) | the same voice's **light set**: one ConvNeXt block fewer, int4. Flash at 0xB00000 (optional). |
| `ito_male_esp32s3{,_int4,_light}.bin` (same sizes, place and license) | the male voice's three sets, flashed to the same three offsets **instead of** the female voice's (`VOICE=male esp32/tools/flash.sh PORT`). The app is the same for both voices. |

`flash.sh` erases the two optional partitions first and writes the sets it finds in `models/`; with only the main set the board behaves as before (no fallback). Flash layout: app 0x10000 (2 MB slot), then three 4.5 MB weight partitions
(`weights` 0x200000, `weights_b` 0x680000, `weights_c` 0xB00000; [`BLOB_FORMAT.md`](BLOB_FORMAT.md)).

At boot the board does these things, in order:
1. It copies the main set to PSRAM and runs the **self-test**: it synthesizes a golden sentence and compares the hash of
   its PCM with the host C engine's. It must print `PASS`.
2. It runs the **board benchmark** (§4) and keeps the fastest weight-staging mode.
3. It runs the **self-calibration** (§4b): it measures the real time of every chunk of a representative sentence with each weight set in turn (main, main-int4, light), keeps the first set whose measured RTF is <= 0.85
   (self-testing it), prints `CALIB` and `TIER_SELECT` lines, and warns and sets a `degraded` flag if even the fastest set measures >= 0.95.
4. It speaks three demo sentences, with the playback start delay planned from the measured chunk times.
5. It prints `READY`.

Serial commands (115200 baud): `say <ids>`, `demo [k]`, `test`, `bench`, `act 8|16`, `wmode 0|1|2|3` (direct, copy, gdma, gdma + prefetch),
`first <frames>` (frames of the first chunk), `delay <ms|auto>` (hold playback until this long after the text arrived; `auto` = the planned delay, the default), `tier [k]` (status, or activate weight set k and self-test it), `status`, `recal` (measure and select again), `simtime <rtf0> <rtf1> <rtf2>|off` (replace the measured chunk times by modelled ones for the next `recal`: timing injection for tests), `unpack pie|c` (int4 weight rows: PIE kernel or C code, same bytes), `stats`, `help`.
The BOOT button repeats the demos. Every `say`/`demo` prints a `TIMING` line with the time to the first chunk, the RTF, the number of **underruns**
(gaps in the speech) and how many GEMM calls had their weights announced ahead (mode 3).

**Rebuild the app** with ESP-IDF 5.5: `esp32/tools/build_fw.sh` writes `prebuilt/ito_app_merged.bin`. The app contains no
model weights (it is GPLv3 code only); the voice model is the separate, non-commercial weights file.

## 2. Try it without a board

```bash
cd esp32/host && make && make test           # host build of the engine; tests against golden references
python3 esp32/tools/chip_wav.py "Any English text." out.wav    # text -> WAV with the chip's exact arithmetic
esp32/tools/build_fw.sh && esp32/tools/qemu/run_qemu.sh       # the real firmware in Espressif QEMU (>= 9.2.2)
```

`make test_sched` runs the boot-calibration and start-delay policy (`engine/itofs_sched.c`, §4b) against a model of the speaker with simulated timings; it needs no weights.

`make test` checks the C engine against an engine-numerics PyTorch reference stored in `host/golden/` (male voice:
`make test VOICE=male`, references in `host/golden/g/`; `VOICE=male esp32/tools/qemu/run_qemu.sh` runs the male voice in QEMU). It checks
durations, the SNR of every stage, and that streaming output is bit-identical to whole-utterance output. It runs for
8- and 16-bit activations, and again under ASan/UBSan. `ito_cli` prints the work done before the first audio chunk
and per second of audio.

## 3. Memory and compute

| | |
|---|---|
| Weights | 3.81 MB blob: 2.37 M int8 + 0.54 M int16 + 0.08 M f32 parameters (the style FiLM is precomputed into a table). The vocoder is 192 wide (576 inner, 5 ConvNeXt blocks); the 4.89 MB blob of 3 October had a 256/768 vocoder (blind test #10: no audible difference, see the root README) |
| Flash | app 379 KB in a 2 MB slot at 0x10000; three 4.5 MB weight partitions at 0x200000 / 0x680000 / 0xB00000 (main 3.81 MB, main-int4 3.20 MB, light 3.05 MB per voice) |
| PSRAM | measured on the board: peak 5.2 of 8 MB PSRAM (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice; on 9 October 2026 the peak was 5372 KB with the female voice and 5380 KB with the male voice, of 8192 KB). Pre-board QEMU figure: peak 5.3 of 8 MB (weights copied to PSRAM + stage ring buffers + PCM chunk buffers + text buffers for 400 tokens) |
| Internal SRAM | measured on the board: peak 354 of 374 KB (with the I2S DMA buffers) (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice; 354 of 374 KB again with the male voice on 9 October 2026). Pre-board QEMU figure: 320 of 379 KB. The firmware checks at boot that the scratch fits in one block with room left over and falls back to PSRAM (slow) if not: look for `internal SRAM before the hot arena` and `arena: hot ... in internal SRAM` in the boot log (this board: hot arena in internal SRAM) |
| Work before the first audio | **23.0-24.3 M instructions and 3.9 MB of weights from PSRAM for the shipped 10-frame (125 ms) first chunk (main int8 set); 16.0-16.5 M and 3.45 MB for a 2-frame one (`first 2`). The same for any sentence length** (forward GRU, fixed style: the text side runs incrementally) |
| Work per second of audio | 263 M int8 MACs + about 1 M f32 MACs + other float work; 76-86 M instructions on the dual-core critical path |
| Weight traffic | 11 MB of weights read from PSRAM per second of audio (one weight pass serves a 24-frame, 300 ms chunk, the two 1202-wide layers two passes; 15 MB with the 256-wide vocoder, 17 MB before that; with 8-frame chunks and the 192-wide vocoder 28 MB, with the 256-wide one 38 MB) |

## 3b. int4 weights (blob format 2)

The ConvNeXt blocks (`pw1`, `pw2`), the wide `harm_proj` and the decoder embed can be stored as **int4**: groups of 32 weights along the input axis, each group with an integer multiplier `m` = 1..7 and a zero point `zp`,
weight `w8 = (q - zp) * m` (the exact layout is in [`BLOB_FORMAT.md`](BLOB_FORMAT.md)). The per-channel float scale is unchanged, so an int4 layer **is** an int8 layer whose weights have this form: same int32 accumulators, same rescale, same audio as the
equivalent int8 blob. Weights are 0.53 bytes each instead of 1 (the main set goes from 3.81 to 3.20 MB). The quantisation is post-training (GPTQ, Frantar et al. 2023, on held-out sentences; no retraining); the exporter is not part of this repository.

*Unpacking.* A packed row is unpacked once per output channel into the aligned scratch row (exactly where an unaligned int8 row is packed today) and the unchanged PIE dot kernel runs on it, so the cost is shared by the 8 to 24 activation rows of a call. Two implementations
produce the same bytes: C on 32-bit words (`itofs_w4_unpack_row`, 2.5 instructions per weight) and a PIE kernel (`itofs_s3_w4_unpack` in `dot_rows_s3.S`, 18 instructions per 32 weights: byte-lane AND, three unsigned 8-bit multiplies and two saturating subtracts on Q registers; none of them overflows
or saturates for the format's value ranges). At boot the firmware compares the PIE kernel with the C code on 7,168 groups covering every `(m, zp)` and uses it only if all bytes agree (`kernels: int4 unpack by the PIE kernel ...`); otherwise it keeps the C code. QEMU's PIE model is not silicon, which is why the check runs on the board too.

**Exactness (all checked, logs in [`results/chip/int4/`](../results/chip/int4)).**
- `python3 esp32/tools/int4_check.py models/<int4 blob>`: for every int4 tensor the engine's int32 accumulators equal an independent NumPy int64 matmul with the unpacked weights; and the int4 blob rewritten as its int8-equivalent blob (`tools/i4_to_i8.py`) gives **bit-identical** PCM through `ito_cli` at 8- and 16-bit activations (self-test and demo sentences). Passes for both voices, main and light set.
- The host test (stage-by-stage SNR against the PyTorch emulation of the int4 numerics, streaming == whole utterance, prefetch hints, ASan/UBSan) passes on the int4 blobs: the emulation's int8 tensors are the unpacked int4 values, so it is the same arithmetic.
- QEMU: the firmware's PCM is bit-identical to the host's in all four weight-staging modes (direct, copy, gdma, gdma + prefetch), one and two cores, 8- and 16-bit activations, both voices.

**Quality** (10 held-out sentences per voice; PESQ-wb against the float PyTorch model; the int4 columns are the main set with int4 blocks):

| | PESQ vs float, int8 | PESQ vs float, int4 | PESQ int4 audio vs int8 audio (mean, min) | log-mel L1 vs float, int8 -> int4 |
|---|---|---|---|---|
| female | 4.50 | 4.48 | 4.52 (4.42) | 0.077 -> 0.132 |
| male | 4.55 | 4.47 | 4.54 (4.47) | 0.071 -> 0.125 |

PESQ moves by 0.02 to 0.08; the mel distance grows by about 75 %, which is large for an objective measure. One listener, in blind test #10 (female, four sentences), rated the int4 emulation 4.00, the same as the int8 engine. The male voice has not been heard.

**Cost, from exact QEMU instruction counts** (female voice, main set, `icprof 2`, same method as §4; the conversion to time is an estimate):

| | before the first chunk (125 ms) | per second of audio, critical path | weights read from PSRAM per second of audio |
|---|---|---|---|
| int8 | 23.0-23.9 M instr, 3.91 MB | 76-86 M instr | 10.5-11.2 MB (+4.5 MB of activation rings) |
| int4, C unpack | 24.8-25.3 M, 3.20 MB | 83-93 M (+8 %) | 8.1-8.8 MB |
| **int4, PIE unpack (shipped)** | **23.4-23.6 M, 3.20 MB** | **78-88 M (+1 %)** | **8.1-8.8 MB (-21 %)** |

So the saving is in memory, not in instructions: with the PIE unpack the extra instructions are about 1 % and the weight traffic falls by a fifth, which matters only where the board is limited by the PSRAM
(the pessimistic column of §4: whole-sentence RTF 0.92-0.93 against 0.97-0.98). Where it is limited by the CPU the int4 set is a hair slower than int8 (optimistic 0.45-0.48 against 0.43-0.47).

## 4. Estimated time to first audio and real-time factor

**Pre-board estimates from exact QEMU instruction counts (the measurement on the board is in §4a).** QEMU runs the real firmware with `-icount shift=0`
(one instruction = one virtual nanosecond), so the instruction counts are exact: per core from text-in to the first chunk,
per second of audio, and on the dual-core critical path (`tools/qemu/icount_profile.sh`, then
`python3 esp32/tools/icount_estimate.py esp32/logs/qemu_icprof.log`). So are the bytes of weights the GEMM staging reads from PSRAM.
Only the conversion to time is an estimate:
- CPU time = critical-path instructions x CPI / 240 MHz, with CPI 1.3 / 1.45 / 1.6 (the range we use for our ASR sister project, Oido);
- PSRAM time = bytes read / effective bandwidth of 80 / 60 / 40 MB/s. The octal PSRAM at 80 MHz DDR is 160 MB/s on paper. Forum measurements
  on the ESP32-S3 (N16R8 and similar) are about 40 MB/s up to about 84 MB/s for reads
  ([1](https://esp32.com/viewtopic.php?p=103811), [2](https://esp32.com/viewtopic.php?p=112434)), 27-52 MB/s for `memcpy` involving PSRAM, and a
  public PIE benchmark shows 10-60 MB/s on other boards ([ESP32-TOPS-BenchMark](https://github.com/nnn112358/ESP32-TOPS-BenchMark)).
  (We read these from search summaries; we could not re-open every forum page.)
- total = max(CPU + (1 - hidden) x PSRAM, PSRAM), where `hidden` is the share of PSRAM time overlapped with compute (GDMA staging):
  1.0 / 0.5 / 0.0 for optimistic / central / pessimistic.

| Exact counts | before the first chunk | per second of audio (critical path) |
|---|---|---|
| first release engine, 100 ms first chunk | 62.5 M instructions, 10.2 MB of weights from PSRAM | 251-271 M instructions, 38 MB |
| previous engine, 25 ms first chunk, 8-frame chunks | 22.9-23.4 M instructions, 4.5 MB | 133-144 M instructions, 38 MB (+ about 4.5 MB of activation rings) |
| previous version of this engine, 2-frame (25 ms) first chunk, then 24-frame (300 ms) chunks | 17.4-17.9 M instructions, 4.8 MB (one pass over the weights) | about 99-101 M instructions, 17 MB (+ about 4.5 MB of activation rings) |
| engine of 3 October, 256-wide vocoder, shipped start-up ramp (10, 11, 12, 14, 18, then 24 frames) | 25.3-26.3 M instructions, 5.1 MB | about 98 M instructions, 15 MB (+ about 4.5 MB of activation rings) |
| **this engine, 192-wide vocoder, same ramp (main int8 set)** | **23.0-24.3 M instructions, 3.9 MB** | **76-86 M instructions, 10.5-11.2 MB** (+ about 4.5 MB of activation rings) |

The counts are the same for short (25 tokens), median (108) and long (175 tokens) sentences. The output is bit-identical between the engines and to the host build.

| Estimated (not measured) | optimistic | central | pessimistic |
|---|---|---|---|
| Time to first audio, first release engine | 338 ms | 463 ms | 672 ms |
| Time to first audio, previous engine (8-frame chunks) | 124-127 ms | 176-179 ms | 266-269 ms |
| Time to first audio, version of 2 October (25 ms first chunk, **gap after it**) | 94-97 ms | 145-148 ms | 237-240 ms |
| Time to first audio, this engine with `first 2` (same idea, **gap after it**) | 87-90 ms | 125-129 ms | 193-197 ms |
| Time to first audio, engine of 3 October (256-wide vocoder), shipped ramp | 137-143 ms | 200-207 ms | 311-318 ms |
| **Time to first audio, this engine (192-wide vocoder), shipped start-up ramp (no gap after it)** | **124-132 ms** | **171-180 ms** | **251-260 ms** |
| Real-time factor, previous engine | 0.72-0.78 | 1.15-1.23 | 1.93-2.04 |
| Real-time factor, version of 2 October (long-sentence limit) | 0.53-0.55 | 0.77-0.79 | 1.18-1.22 |
| Real-time factor, engine of 3 October (256-wide vocoder; long-sentence limit) | 0.53 | 0.76 | 1.14 |
| **Real-time factor, this engine** (long-sentence limit) | **0.47** | **0.65** | **0.96** |
| Real-time factor, this engine, whole sentences including the start-up ramp | 0.43-0.47 | 0.63-0.66 | 0.97-0.99 |
| Start delay for gapless playback, version of 2 October | about 220 ms | 342-346 ms | 850-2700 ms |
| Start delay for gapless playback, engine of 3 October (256-wide), shipped ramp | 137-143 ms | 215-240 ms | 880-2200 ms |
| **Start delay for gapless playback, this engine, shipped ramp** | **125-130 ms (= time to first audio)** | **176-182 ms** | **558-663 ms** (short sentences need the most; this column is a hair under RTF 1 and is not a guarantee) |

(ranges: the self-test sentence and the three demo sentences, 25 / 84 / 108 / 175 tokens, both voices. The 192-wide figures were recomputed from fresh QEMU instruction counts of the final blobs of both voices.)

Then I2S adds at most one 20 ms DMA buffer.

### The three weight sets (same method, final firmware, both voices)

Exact counts and estimates for each set that the boot calibration can choose from (§4c); `icprof 2` on the final firmware, both voices, the self-test sentence and the three demo sentences (25 / 84 / 108 / 175 tokens), ranges over all of them.
Pre-board **estimates** (the measured numbers for both voices are in §4a; the int4 and light sets were timed by hand only); same CPI, bandwidth and overlap assumptions as above.

| | main int8 (set 0) | main int4 (set 1) | light, 4 blocks int4 (set 2) |
|---|---|---|---|
| blob per voice | 3.81 MB | 3.20 MB | 3.05 MB |
| before the first audio (125 ms chunk): instructions on the critical path, weights from PSRAM | 23.0-24.3 M, 3.91 MB | 23.4-24.5 M, 3.20 MB | 21.8-22.9 M, 3.07 MB |
| per second of audio: critical-path instructions, weights from PSRAM (+ about 4.5 MB of activation rings) | 76-86 M, 10.5-11.2 MB | 78-88 M, 8.1-8.8 MB | 73-83 M, 7.7-8.4 MB |
| **time to first audio** (optimistic / central / pessimistic) | **124-132 / 171-180 / 251-260 ms** | **127-133 / 168-174 / 236-243 ms** | **118-124 / 157-164 / 222-229 ms** |
| **RTF, whole sentences with the ramp** | **0.43-0.47 / 0.63-0.66 / 0.97-0.99** | **0.45-0.48 / 0.62-0.64 / 0.92-0.93** | **0.42-0.45 / 0.59-0.61 / 0.88** |
| RTF, long-sentence limit (175 tokens) | 0.47 / 0.65 / 0.96 | 0.48 / 0.64 / 0.91-0.92 | 0.45 / 0.61 / 0.87 |
| **start delay for gapless playback** | **125-130 / 176-182 / 558-663 ms** | **127-133 / 173-179 / 481-526 ms** | **118-124 / 162-169 / 406-429 ms** |
| structural model, long-sentence RTF: nothing overlapped / `wmode 2` / `wmode 3` (optimistic; central; pessimistic) | 0.66, 0.78, 0.96-0.97 / 0.59, 0.69-0.70, 0.86-0.87 / 0.54-0.55, 0.64, 0.77-0.78 | 0.64, 0.75, 0.91-0.92 / 0.58, 0.68, 0.82 / 0.54, 0.62, 0.74 | 0.61, 0.71, 0.87 / 0.55, 0.64, 0.78 / 0.51, 0.59, 0.70 |

What this says, without rounding in our favour:
- **int4 buys memory, not instructions**: with the PIE unpack the critical path grows by about 1 % and the weight traffic falls by a fifth, so set 1 is a little slower than set 0 when the board is CPU-bound (optimistic 0.45-0.48 against 0.43-0.47) and clearly faster when it is PSRAM-bound (pessimistic 0.92-0.93 against 0.97-0.99). With the C unpack
  (before the PIE kernel) set 1 was no faster than set 0 even in the pessimistic column.
- **The light set buys another 4-5 %**: one block fewer is about 6 % fewer instructions and 5 % less traffic than set 1. In the pessimistic column it is the only set below 0.9 (0.88) and the only one whose gapless delay is under half a second for every test sentence.
- **None of this makes the pessimistic column comfortable.** There, with a 0.85 target, the boot would find no set at or below it, keep the light set as `MARGINAL` (0.88), and plan a start delay of about 410 ms. A board that is more than about 8 % slower than the pessimistic assumption would measure the light set at 0.95 or more and report `degraded`.
  In the structural model, which hides PSRAM time behind compute the way `wmode 3` is built to, even set 0 is at 0.77-0.78 in the pessimistic column.

| if the board behaves like the ... column, the boot would | optimistic | central | pessimistic |
|---|---|---|---|
| measure (sets 0 / 1 / 2, whole sentences) | 0.43-0.47 | 0.63-0.66 | 0.97-0.99 / 0.92-0.93 / 0.88 |
| keep | set 0 | set 0 | set 2, `MARGINAL` (nothing reaches 0.85) |
| plan a gapless start delay of | 125-130 ms | 176-182 ms | 406-429 ms |



Reading this honestly:
- **Time to first audio is not the same as gapless speech, so the engine now starts with a ramp.** Every chunk costs one pass over the weights plus about
  6.5 ms (central) per frame; one frame is 12.5 ms of audio. A first chunk of 2 frames is out after about 125 ms, but it holds only 25 ms of audio and the
  24-frame chunk behind it needs about 180 ms more: the speech has a gap of about 150 ms (the version of 2 October needed a 340 ms start delay
  against it). The shipped schedule instead makes the first chunk 10 frames (125 ms of audio), then 11, 12, 14, 18 and 24, with the text side advancing in 8-token
  steps meanwhile (one 24-token step would make a single chunk 65 ms slower than its neighbours): each chunk is then ready before the audio of
  the chunks before it has played out, so playback can start with the first chunk. The cost is a later first sound (171-180 ms central instead of 125-129). In the
  central case the speech is gapless from 176-182 ms after the text arrived (the first chunk plus 2-5 ms, spent on the chunks that carry a text step);
  `delay <ms>` holds playback for that long. In the optimistic case the first chunk is already gapless. In the pessimistic case the whole-sentence RTF is 0.97-0.99 for
  sentences of 84-175 tokens and 1.05 for the 25-token one (the ramp is a larger share of a short sentence), so the start delay is 560-660 ms and a very slow board would still have gaps (§4c replaces this assumption with a measurement).
  `first 2` restores the old lowest-latency start (first sound at 125-129 ms central, with the gap). The audio is bit-identical for every schedule (host test G, QEMU).
- **The pessimistic case is only just under real time.** At 40 MB/s of PSRAM bandwidth with no overlap and CPI 1.6 the estimate is RTF 0.96 (long sentences; it was 1.14 with the 256-wide vocoder),
  and 0.97-0.99 for whole sentences with the ramp. That is not a margin. The CPU alone is 0.46 / 0.52 / 0.57 of real time (optimistic / central / pessimistic); the PSRAM bandwidth assumption decides the rest. This column is defined as
  "nothing overlapped", so no change to the staging can improve it; only fewer weight bytes or fewer instructions can.
- **What the central column assumes, and a structural check.** The central estimate hides half of the PSRAM time behind compute. To see what the firmware's
  staging can actually do, `tools/trace_sim.py` replays the exact per-call trace of a chunk (every GEMM call and every row-parallel float loop, with its exact
  instruction count per core and weight bytes) through a model of the weight DMA: both cores share one PSRAM bus, each has two staging buffers. Still an estimate, but without the hidden-share knob:

  | long-sentence RTF (24-frame chunks) | optimistic (CPI 1.3, 80 MB/s) | central (1.45, 60) | pessimistic (1.6, 40) |
  |---|---|---|---|
  | nothing overlapped | 0.66 | 0.78 | 0.96-0.97 |
  | GDMA double buffer inside each GEMM call (`wmode 2`, the staging of the version of 2 October) | 0.59 | 0.69-0.70 | 0.86-0.87 |
  | **plus cross-call prefetch (`wmode 3`)**: the engine announces the next call's weights, which are fetched during the float work in between | **0.54-0.55** | **0.64** | **0.77-0.78** |

  (Recomputed for the 192-wide vocoder with `tools/trace_table.py` on the 175-token sentence; the same script on the 3 October log reproduces the earlier 0.78 / 0.92 / 1.14, 0.67 / 0.80 / 1.01, 0.63 / 0.74 / 0.91.)

  So the central column's assumption (half hidden) is about what `wmode 3` reaches (RTF 0.64 against the column's 0.65); `wmode 2` hides less (RTF 0.69). With the prefetch the gapless
  start delay is 167-175 ms central and the time to first audio the same (same model; 143-150 ms for both, optimistic). The same model says the bus, not the CPU, limits the GEMM calls (the two cores together ask
  for about 110 MB/s while computing a tile), and that 4 KB tiles lose about 2 % of the central chunk time (and about 12 ms of start delay) against 10 KB ones. If the DMA does not behave like this on the chip, the board benchmark
  chooses another mode; it also compares the audio of every mode with the direct one and drops a mode that differs.
- **What changed on 3 October.** Cross-call weight prefetch (`wmode 3`), the start-up ramp with small text steps, 12-row passes for the two 1202-wide layers
  (their weights are read twice per chunk instead of three times: -12 % traffic, paid for by 4 KB instead of 10 KB staging tiles), a 4-channel depthwise kernel
  (-3 % instructions): all bit-identical. What it did not change: the pessimistic column is still above 1.
- **Correction history.** Our first public README said 130-210 ms to first audio and RTF 0.58-0.92. Those numbers counted only int8 MACs at an assumed
  0.5-1 GOPS and were too optimistic. Counting every instruction and the PSRAM traffic, the first release engine came to 338 / 463 / 672 ms and RTF about 2,
  and the second version to 176-179 ms and RTF 1.15-1.23 (not real time). This version halves the weight traffic with 24-frame chunks (a leaner scratch
  layout makes them fit in internal SRAM) and removes about 30 % of the critical-path instructions (fused epilogues, packed weight rows, constant tables).
  The model of time is unchanged: CPI 1.3 / 1.45 / 1.6, PSRAM 80 / 60 / 40 MB/s, 1.0 / 0.5 / 0.0 of the PSRAM time hidden behind compute.
- **What changed on 5 October.** The vocoder is now 192 wide (576 inner) instead of 256 (768): -22 % weights, -24 % MACs per second of audio, no engine change (the widths are read from the blob).
  Blind test #10 (one listener, four sentences per system, female voice) rated the reference 4.25, the 256-wide vocoder 4.00, the 192-wide one 4.00 and the 192-wide one with int4 weights 4.00: no difference he could hear.
- **Real time was measured on one board (§4a).** The first audio chunk is ready in under 200 ms and synthesis runs faster than real time on this board (RTF 0.66), with 0 underruns when playback starts after the planned 246 ms delay. This is one board: the female voice in four boots (8 October and three on 9 October, all resets) and the male voice in two boots, with the main int8 set chosen by the boot calibration, plus hand-timed runs of the other two sets; other boards, cold power-on boots, the DAC stage and listening are not measured. We will report more boards as they are measured.  If a board turns out slower, the fallbacks are int4 weights in the ConvNeXt blocks and fewer blocks (the int4 and light sets ship as fallback weight sets).

**What the board prints** (please send the whole log from reset to `READY`):

```
SELFTEST dual-core, 8-bit activations: ... fnv32 a7d804ab (host a7d804ab) -> PASS
BENCH_BW psram->sram memcpy: X MB/s | gdma: X MB/s | psram read through the data cache: X MB/s
BENCH_GEMM <layer shape> rows 8 | direct 1c/2c | copy 1c/2c | gdma 1c/2c  GMAC/s
BENCH_TTS wmode <m> demo k: tokens, audio s | text side ms | first chunk ms | total ms, RTF | GEMM GMAC/s, non-GEMM ms
BENCH_TTFA_MODEL wmode <m>: first chunk = a ms + b ms/token
BOARD_SUMMARY ... best_wmode=... mean first chunk, mean RTF, GEMM GMAC/s per mode
```

These lines give the effective G, the real float-work time, the PSRAM bandwidth and the time to first audio directly.

## 4a. Measured on the board (8 and 9 October 2026)

**Measured on a board** (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice; the 9 October additions follow the table): with the main int8 set the first audio chunk (125 ms of audio) was computed and ready 184 ms after the text was handed to the engine (183.9–184.3 ms over the three demo sentences), and the real-time factor was 0.66 on the boot calibration sentence (0.660–0.674 on the demo sentences). Gap-free playback is a separate number: the firmware plans to start playback about 246 ms after the text arrives, and with that planned delay the three demo sentences had 0 underruns. No DAC or amplifier was attached and no audio was listened to, so the I2S/DAC output stage is not included, these are not acoustic measurements, and the underrun count is the firmware's own accounting against its playout clock. The first audio chunk is ready in under 200 ms and synthesis runs faster than real time on this board (RTF 0.66), with 0 underruns when playback starts after the planned 246 ms delay. For the female voice, the other two weight sets were timed by hand once on 8 October with the `tier` serial command (one run each, not chosen by the boot calibration): main int4 set RTF about 0.65, first chunk about 180 ms; light set RTF about 0.61, first chunk about 168 ms. **Update, 9 October 2026** (same board, same firmware fedbd26, still no DAC and nothing listened to). Male voice: the boot calibration chose the main int8 set (RTF 0.659 on the calibration sentence, in both of two boots), the bit-exact self-test passed (137,400 samples, identical to the host C engine), the first audio chunk was ready in 183.8–184.8 ms over the demo sentences, the demo-sentence RTF was 0.659–0.672, the firmware planned a gap-free start 248–249 ms after the text arrives, and underruns were 0. The male voice's other two sets were timed by hand with `tier` (not boot-chosen; the firmware plans no start delay for hand-timed sets), demo sentences only: main int4 set first chunk 179.5–179.8 ms, RTF 0.645–0.653 (two passes); light set first chunk 167.9–168.0 ms, RTF 0.606–0.613 (one pass); the self-test passed with each set. Female voice, repeats: the same image booted four times in all (8 October and three times on 9 October, each after a USB-serial or watchdog reset, not a cold power cycle); the first chunk was ready in 183.5–184.5 ms over all demo runs, the boot-calibration RTF was 0.660–0.661, the planned start was 246–248 ms, the bit-exact self-test passed every time and underruns were 0. After a demo the PSRAM peak was 5372 KB (female) and 5380 KB (male) of 8192 KB, and internal SRAM 354 of 374 KB (stack high-water marks were not printed). All of this comes from one board: the repeats and the male runs show repeatability on one chip, not a second board.

| | measured | pre-board estimate for this set (optimistic / central / pessimistic) |
|---|---|---|
| Real-time factor (boot calibration, 84-token sentence) | **0.66** | 0.43–0.47 / 0.63–0.66 / 0.97–0.99 |
| Real-time factor, demo sentences (25 / 108 / 175 tokens) | 0.674, 0.660, 0.662 | |
| First audio chunk (125 ms of audio) computed and ready after the text is handed to the engine; excludes the I2S/DAC stage | **184 ms** (183.9–184.3 ms over the demo sentences) | 124–132 / 171–180 / 251–260 ms |
| Playback start delay planned by the firmware for gap-free playback | about 246 ms (246–247 ms) | 125–130 / 176–182 / 558–663 ms |
| Underruns in the demo sentences, with that planned delay (firmware accounting; no audio output) | 0 | 0 expected |
| PSRAM to SRAM copy / GDMA / cached reads (boot benchmark) | 86.0 / 36.4 / 89.2 MB/s | 40-80 MB/s assumed for the copy |
| Fastest weight-staging mode (`BOARD_SUMMARY`) | `direct`: mean first chunk 184.4 ms, mean RTF 0.664, GEMM 1.060 GMAC/s | the central column assumed `wmode 3` |
| Main int4 set (female), timed by hand once (`tier 1`), not boot-chosen | RTF about 0.65, first chunk about 180 ms | |
| Light set (female), timed by hand once (`tier 2`), not boot-chosen | RTF about 0.61, first chunk about 168 ms | |

Boot calibration (`TIER_SELECT`), 8 October, female voice: chose the main int8 set with a measured RTF of 0.66 (status OK, degraded flag 0; main int4 and light were not needed, so the boot did not time them; they were timed by hand afterwards).

**9 October 2026: the male voice, the other sets and repeat boots** (same board, same firmware `fedbd26`, still no DAC and nothing listened to).

| | measured |
|---|---|
| Male voice, boot calibration (two boots) | chose the main int8 set (set 0), RTF 0.659 on the calibration sentence in both, status OK, degraded flag 0 |
| Male voice, bit-exact self-test | PASS: 137,400 samples, fnv32 35f96c0b, identical to the host C engine (also PASS with the hand-timed sets: set 1 b46d58de, set 2 a14d1c89) |
| Male voice, demo sentences (25 / 108 / 175 tokens), main int8 set | RTF 0.672, 0.659, 0.664; first audio chunk ready in 183.8–184.8 ms; planned start delay 248–249 ms; 0 underruns |
| Male voice, main int4 set, timed by hand (`tier 1`, two passes), not boot-chosen | demo sentences: RTF 0.645–0.653, first chunk 179.5–179.8 ms; the firmware plans no start delay |
| Male voice, light set, timed by hand (`tier 2`, one pass), not boot-chosen | demo sentences: RTF 0.606–0.613, first chunk 167.9–168.0 ms; the firmware plans no start delay |
| Female voice, repeats: the same image booted four times (8 October and three times on 9 October; resets, not cold power cycles) | first chunk ready in 183.5–184.5 ms over all demo runs; boot-calibration RTF 0.660–0.661; demo-sentence RTF 0.659–0.674; planned start delay 246–248 ms; self-test PASS every time (fnv32 023bd6b1, identical to the host); 0 underruns |
| Memory after a demo | PSRAM peak 5372 KB (female) / 5380 KB (male) of 8192 KB; internal SRAM 354 of 374 KB (both); stack high-water marks were not printed |
| Boot benchmark, all boots | PSRAM-to-SRAM copy 86.0 MB/s, GDMA 36.3–36.4 MB/s, cached reads 89.2 MB/s; fastest staging mode `direct` |

The male voice and the repeats are the same single board, so they show repeatability on one chip, not a second board; none of the boots was a cold power cycle (the reset reason in every log is `USB_UART_CHIP_RESET`, not a power-on reset). The hand-timed sets were typed in by hand in one session: the male int4 set was timed in two passes and the light set in one. Two earlier male `tier` logs are kept as `*_ABORTED_*` because commands overlapped the reboot after the port was opened; they are not used for any number. The logs are in [`logs/`](logs/README.md).

Against the pre-board estimates (optimistic / central / pessimistic: RTF 0.43–0.47 / 0.63–0.66 / 0.97–0.99, first audio chunk 124–132 / 171–180 / 251–260 ms), the measured RTF of 0.66 is at the top of the central estimate and far from the pessimistic one (0.97–0.99), and the measured first chunk of 184 ms is slightly above the central estimate and below the pessimistic one. In the boot benchmark the PSRAM-to-SRAM copy ran at 86.0 MB/s (above the 40–80 MB/s the estimates assumed), GDMA at 36.4 MB/s (slower than a plain copy), cached PSRAM reads at 89.2 MB/s. The firmware therefore stages weights with `direct` reads, the fastest of the four modes it benchmarks (mean RTF 0.66, GEMM 1.060 GMAC/s). Still not measured: audio through a DAC and any listening to the output (no DAC was attached), a second board or a second chip revision, cold power-on boots (every boot was a reset), power draw, long-run thermal behaviour and supply sensitivity, stack high-water marks (not printed), the male voice's sound quality through the chip (its audio is bit-identical to the host engine: self-test PASS), and a planned start delay for the hand-timed sets (the firmware plans none). Still estimates: the optimistic, central and pessimistic columns, which are the pre-board predictions kept for comparison.

**What 'first audio' means here.** The time from the text being handed to the engine until the first audio chunk (125 ms of audio) is computed and ready in memory. No DAC or amplifier was attached and no audio was listened to, so the I2S/DAC output stage is not included, these are not acoustic measurements, and the underrun count is the firmware's own accounting against its playout clock. Gap-free playback is a separate number: the firmware plans to start playback about 246 ms after the text arrives, and with that planned delay the run had 0 underruns. We do not claim that speech is audible or gap-free at 184 ms.

Method: one board and the fixed prebuilt firmware (commit fedbd26) flashed as in §1; on 8 October one boot (reset reason `USB_UART_CHIP_RESET`, not a power-on reset), then the `tier` command by hand; the 9 October runs are described under the second table; the numbers are the ones the firmware prints (`BOARD_SUMMARY`, `CALIB`, `TIER_SELECT`, `TIMING`). Notes from the person who ran it: First run on silicon; native USB only, no DAC/amp attached (audio not heard). Bench scratch limited to 8 GEMM rows (free internal SRAM fragmented). Underruns counted against the DAC playout clock. The full boot log of 8 October is in the repository as `esp32/logs/board_fixed_boot_and_tiers_2026-10-08.log`; all logs, with a note on which is which, are listed in [`esp32/logs/README.md`](logs/README.md).

### Stability on the board (9 October 2026)

Same board and firmware (`fedbd26`), 240 MHz, female voice, USB powered, nothing attached, I2S idle. **Soak:** each iteration runs the boot golden sentence in three modes (dual-core 8-bit, single-core, dual-core 16-bit) and compares the output hash with the host C engine. Main int8 set: 300 iterations (900 runs, all PASS, 0 FAIL, 78 minutes; hashes `023bd6b1` and `9666bb9e`, identical to the host C engine). Int4 set: 100 iterations (300 PASS, 0 FAIL; host hash `363d7d98`). Light set: 100 iterations (300 PASS, 0 FAIL; `5f12d5e1`). Main int8 set again with I2S idle: 100 iterations (300 PASS, 0 FAIL). In all, 600 iterations (1,800 runs) in about 2.5 hours, with no panic, watchdog or reboot in any phase. A 175-token demo repeated 15 times in the main phase gave a first chunk of 183.9–184.6 ms and an RTF of 0.662–0.663, with no drift and no sign of throttling. **Synthetic stress:** Ito's own PIE kernel on both cores in lock-step at 240 MHz for about 21 minutes (3 rounds, random int8 data, every call compared with a one-core reference): K=704 (64 rows, 45 KB shared tile, 88/88 split, the pattern reported by the Oído project), K=576, K=192, K=1216 and a one-core control gave 0 mismatches and 0 crashes. **What this says, and no more:** no corruption or crash was observed in 600 golden-sentence iterations and a 21-minute synthetic stress on one board. **The Oído finding.** The [Oído](https://github.com/lokutor-ai/oido) project reports (["Dual core"](https://github.com/lokutor-ai/oido#dual-core)) that on silicon its two-core mode gives wrong output or crashes when both cores run PIE kernels at the same time; the case described to us was a two-core 240 MHz int8 K=704 streaming kernel that was exact at 160 MHz. Ito did not reproduce that with its own kernel on this board. That is board- and pattern-specific information, not proof that the problem is absent elsewhere: our stress started core 1 by spinning on a flag and the engine starts it with a FreeRTOS notification, a start with both cores aligned to the same cycle was not tested, Ito's real tiles are 5–15 KB, no real layer has K=704, and about 60 % of the time goes to float work between the GEMM calls (see [`esp32/logs/dualcore_load_comparison.md`](logs/dualcore_load_comparison.md)). Anyone running dual-core PIE streaming at 240 MHz should test it, and on more than one board. **Not covered:** a second board or chip revision, cold power-on boots, supply measurements (the USB voltage was not measured), power draw, temperature, the male voice (not soaked), audio output (no DAC) and long-term (days) stability. Logs: [`esp32/logs/`](logs/README.md); stress source: [`esp32/tools/stress/`](tools/stress/README.md).

## 4b. Boot calibration and start-delay policy

`engine/itofs_sched.c` is the policy that turns **measured** chunk times into decisions. It has no clock of its own: the firmware feeds it microseconds, the tests feed it simulated ones.
- *Measured RTF of a trace* (one representative sentence synthesised chunk by chunk with the shipped schedule): the larger of the median full-chunk figure and the whole-trace figure (ramp included).
- *Choice among weight sets* (best quality first): the first whose measured RTF is <= 0.85; otherwise the lowest, reported as marginal (0.85 to 0.95) or **degraded** (>= 0.95).
- *Start delay per utterance* (an adaptive jitter pre-buffer): a model of the speaker and of the 6 PCM buffers, driven by the measured chunk times times a 1.25 safety factor (a trace's own chunks, then its steady median, and a fixed-plus-per-frame fit for the short last chunk),
  returns the smallest delay with no predicted underrun and never less than the first chunk's production time; the chunk times actually seen in the utterance replace the model's as they arrive.

`make test_sched` (host) checks it with simulated timings against an independent discrete-event model of the speaker (true chunk times of a *different* sentence with its own jitter): the weight-set choice at the exact thresholds and with missing sets;
**8,640 random sentences** (RTF 0.30 to 0.85, three splits of fixed and per-frame cost, 3 to 400 tokens, +-4 % jitter per chunk and +-4 % sentence-to-sentence cost between calibration and playback) with **0 underruns**; the first audio released with no extra wait on a fast board;
a board slower than real time reported infeasible rather than hidden; slower-than-calibrated chunks seen early raising the delay. Outside the margin (+-8 % and +-8 %) 4 of 2,880 sentences underrun, which the test reports as information, not a guarantee.
The firmware integration is §4c; its first run on a board is reported in §4a.

## 4c. Weight sets and the boot check

The board carries up to three weight sets per voice, best quality first, each a complete blob in its own flash partition (`firmware/partitions.csv`, [`BLOB_FORMAT.md`](BLOB_FORMAT.md)):

| set | flash | what | file (Hugging Face) |
|---|---|---|---|
| 0, main | `weights` at 0x200000 | 192/576 vocoder, 5 blocks, int8 (the reference-quality set; bit-exact with the PyTorch emulation) | `ito_<voice>_esp32s3.bin` |
| 1, main-int4 | `weights_b` at 0x680000 | the same model with int4 weights in the ConvNeXt blocks, `harm_proj` and the decoder embed | `ito_<voice>_esp32s3_int4.bin` |
| 2, light | `weights_c` at 0xB00000 | 192/576 vocoder with 4 blocks (one block fewer, fine-tuned), int4 weights as set 1 | `ito_<voice>_esp32s3_light.bin` |

**What happens at boot.** Only one set is in PSRAM at a time. The firmware activates the sets in that order and, for each, synthesises a representative sentence (the blob's own
self-test sentence, 84 tokens, about 5.6 s of audio) without playback, with the shipped chunk schedule, and records the **wall time of every chunk** (compute plus the weight fetch from PSRAM).
From the trace it computes a measured real-time factor (the larger of the median full-chunk figure and the whole-sentence figure, ramp included), and keeps **the first set whose measured RTF is <= 0.85**. If none
reaches 0.85 it keeps the one with the lowest RTF and prints a `NOTE` (0.85 to 0.95, "marginal") or, if even that one is >= 0.95, a `WARNING: DEGRADED` line and sets a `degraded` flag (also shown by `status`). The chosen set is
then self-tested against the host engine's PCM hash like every set (a set that fails is excluded and the next one is tried).
`TIER_SELECT` is the line to send us:

```
CALIB set 0 ('weights', vocoder 192/576 x5, int8): 84 tokens, 21 chunks | first chunk 139.0 ms (begin 3.1 ms) | steady chunk 190.2 ms of 300 ms audio: RTF steady 0.634, whole 0.672 -> measured RTF 0.672 | weight staging gdma+prefetch
TIER_SELECT chosen=0 ('weights', vocoder 192/576 x5, int8) measured_rtf=0.672 target=0.85 status=OK degraded=0 | measured: set0 0.672 set1 not needed set2 not needed
```

**Start delay, per utterance.** The same measured chunk times drive an **adaptive jitter pre-buffer**: after the first chunk of every utterance the firmware models the speaker (chunk k can start playing when chunk k - 1 has finished and chunk k has
been produced; production of chunk k waits for the PCM buffer of chunk k - 6) with the measured chunk times times a 1.25 safety factor, and releases the first audio at the smallest delay after which the model predicts no underrun
(never earlier than the first chunk is ready). While the player waits, the times actually seen in this utterance replace the model's, so a sentence that turns out slower than calibrated raises the delay before playback starts. `delay <ms>` fixes it by hand, `delay auto` returns to the plan.
The `TIMING` line reports the planned delay and the underruns.

**What this does and does not promise.** If the chunk times measured at boot are representative of later sentences (to within the 25 % margin), then for a set with a measured RTF below the target playback is gapless, with the
smallest start delay the model allows. It is a check **by measurement, on the board that makes it**: the margin, the 0.85 target and the representativeness of the calibration sentence are engineering choices that a real board can contradict
(a second core busy with something else, a thermal throttle, PSRAM contention from a radio). It cannot make a slow board fast: with no fallback below 0.95 the firmware says so instead of stuttering silently. It has run on one board (§4a; four female and two male boots, each time choosing the main int8 set): the 8 October boot calibration chose the main int8 set with a measured RTF of 0.66 (status OK, degraded flag 0; main int4 and light were not needed, so the boot did not time them; they were timed by hand afterwards); whether the margin and the 0.85 target hold on other boards, supplies and temperatures is not tested.

**How it is tested without a board.**
- `make test_sched` (host): the policy (`engine/itofs_sched.c`, no clock of its own) against a discrete-event model of the speaker with **simulated timings**: 10 weight-set choices (thresholds at exactly 0.85 and 0.95, missing sets); 8,640 random sentences
  (RTF 0.30 to 0.85, three ratios of fixed to per-frame cost, 3 to 400 tokens, +-4 % jitter per chunk and +-4 % sentence-to-sentence cost between calibration and playback): 0 underruns; the first audio is released with no extra wait when the board is fast;
  a board slower than real time is reported infeasible, not hidden; observed slow chunks raise the delay.
- QEMU (`simtime <rtf0> <rtf1> <rtf2>` then `recal`): the firmware loads all three sets in turn from flash, runs the real selection with modelled chunk times and the real self-test of every chosen set. Logged
  in [`results/chip/qemu_sets_female.log`](../results/chip/qemu_sets_female.log) and `_male`: modelled RTFs 0.90 / 0.80 / 0.70 pick set 1 (measured 0.94 / 0.84 including the ramp); 0.95 / 0.92 / 0.90 pick set 2 as `MARGINAL`; 1.10 / 1.05 / 0.97 pick set 2 as `DEGRADED` with the warning and `degraded=1`; switching back to the real (QEMU) clock picks set 0.
  Nine PCM dumps per run, all bit-identical to the host's. [`qemu_modes_*.log`](../results/chip/): each set in the copy, gdma and gdma + prefetch staging modes and with the C and the PIE int4 unpack, eleven dumps per voice, all bit-identical.

## 5. Numerics, and what was verified

**Precision plan:**
- **F0 path** (text encoder, GRU input matrices, projection, duration head, prosody net): int16 weights × 15-bit
  activations.
- **Mel head and vocoder**: int8 per-channel weights × 8-bit per-frame activations. Every 16-bit-activation plan
  scored the same PESQ, so the cheapest one ships.
- **Float32**: LayerNorms, depthwise convolutions, GRU recurrence, FiLM, output heads, harmonic source, FFTs.
- The engine has its own transcendental functions built from + − × ÷ only. With `-ffp-contract=off`, the host and the
  chip produce bit-identical PCM.

| check | result |
|---|---|
| engine-numerics reference (float mode) vs the PyTorch model | mel 123 dB, prosody 138 dB SNR, durations exact: the reference *is* the model |
| **quantised (engine numerics) vs float PyTorch, 10 held-out sentences, 192-wide vocoder** | **PESQ-wb 4.50 female / 4.55 male** (median 4.54 / 4.56; 3-s segments 4.57 / 4.56, min 4.49 / 4.48; one female sentence scores 4.04 on whole-utterance PESQ, an alignment artefact: its 3-s segments are 4.49+); durations equal on 10/10; log-mel L1 0.077 / 0.071. For scale: float against float with a different noise draw scores 3.82 / 3.49. (The 256-wide vocoder scored 4.52 on 20 sentences.) |
| C engine vs reference, stage by stage | durations exact; text side 89–91 dB, durations 98 dB, F0 95 dB; mel head 57–59 dB (8-bit) / 87 dB (16-bit); vocoder spectrum 46–47 dB (8-bit) / 80–81 dB (16-bit) |
| why the 8-bit SNRs are lower | the reference against **itself** with 1e-7 relative noise gives the same range: an 8-bit quantiser flips a whole step when its input crosses a rounding boundary. The C engine sits inside the reference's own rounding noise |
| C engine audio, PESQ | against float 4.37–4.60 over the three golden sentences of both voices (female 4.56 / 4.37 / 4.52, male 4.59 / 4.53 / 4.60), against the reference 4.38–4.62 |
| streaming == whole utterance | **bit-identical** for chunks of 1, 2, 3, 5, 8, 16, 24 and 32 frames, for the shipped start-up ramp and for text steps of 1, 3, 8, 13, 24 and 40 tokens (host test G); incremental and whole-sentence text side bit-identical |
| weight-prefetch hints | host test H: the engine announces the weights of the next GEMM call before the current one, for every call of every test sentence (0 unannounced, 0 mismatched, chunks of 2, 8, 24 frames) and the audio does not change |
| ASan / UBSan | clean |
| **QEMU** (Espressif 9.2.2, quad PSRAM, I2S compiled out) | self-test PASS; full PCM bit-identical to the host (female 134100/134100 samples, male 137400/137400) in all four weight-staging modes (direct, copy, gdma, gdma + prefetch; QEMU has no GDMA, so the two GDMA modes run the same control flow with a software copy that is only done when the tile is waited for and poisons the buffer until then); self-test PASS on 1 and 2 cores, 8- and 16-bit activations, both voices ([`female`](../results/chip/qemu_rtf_female.log), [`male`](../results/chip/qemu_rtf_male.log); the earlier direct/copy staging runs are in [`qemu_v3.log`](../results/chip/qemu_v3.log)) |
| int4 weights (§3b) | `int4_check.py` passes on all four int4 blobs (C GEMM == int64 reference; int4 blob == int8-equivalent blob, bit-identical PCM at 8/16-bit); host test + ASan/UBSan pass; firmware PCM bit-identical in QEMU in all staging modes, both voices ([`results/chip/int4/`](../results/chip/int4)) |
| weight sets and boot selection (§4c) | QEMU, all three sets of both voices from flash: every set's firmware PCM is bit-identical to the host's, in the four staging modes and with the C and the PIE int4 unpack ([`qemu_modes_*.log`](../results/chip/)); the selection and degraded logic runs with injected timings ([`qemu_sets_*.log`](../results/chip/)); `make test_sched`: 8,640 simulated sentences, 0 underruns |

Timings printed under QEMU are emulator wall-clock times and say nothing about the chip.

## 6. Not verified yet

- **On silicon, one board is measured (§4a)** (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), firmware fedbd26): the female voice on 8 October 2026 and in three more boots on 9 October, the male voice in two boots on 9 October; every time the firmware booted, passed its bit-exact self-test against the host engine, and printed the numbers above. GDMA staging: the on-board GDMA self-test passed; int4 unpack: PIE kernel (its on-board check against the C code passed). The first run also found and fixed four first-silicon bugs (a GDMA self-test heap overflow that crash-looped the first public prebuilt image, bench scratch fragmentation, underrun accounting, USB console input); see the commits and the note at the top. Not measured or not verified: the audio quality through a DAC (none was attached, nothing was listened to), the I2S output stage latency, power draw, the BOOT button, a second board or chip revision, cold power-on boots (every boot was a reset), long-run thermal behaviour and supply sensitivity, stack high-water marks (not printed), the male voice's sound quality through the chip (its audio is bit-identical to the host engine: self-test PASS), and the int4 and light sets in the boot calibration (they were timed by hand only).
- **The boot self-calibration and the start-delay plan** ran on this one board (four female boots and two male boots): the boot calibration chose the main int8 set with a measured RTF of 0.66 (status OK, degraded flag 0; main int4 and light were not needed, so the boot did not time them; they were timed by hand afterwards; 0.660–0.661 in the four female boots, 0.659 in both male boots). Whether the 1.25 margin covers the variation between the calibration sentence and later ones is only partly tested (every demo sentence of all six boots had 0 underruns with the planned delay).
- **Stability is observed, not proven.** No corruption or crash was seen in 600 golden-sentence iterations and a 21-minute synthetic two-core stress on one board (see "Stability on the board" above). Not covered: a second board or chip revision, cold power-on boots, supply measurements, power draw, temperature, the male voice, audio output and runs of days. The Oído project reports faults with concurrent two-core PIE kernels at 240 MHz on this chip; Ito did not reproduce them, but test on more than one board.
- **Most of the float-work cycle costs behind the pre-board estimates** are still assumptions; the board's `BENCH_*` lines measure some of them.
- No formal listening test compares the quantised chip output with the float model. The PESQ of 4.52 against float
  suggests the difference is small.

## Layout

```
engine/      itofs.c / itofs.h: portable C99 engine (also reads the older arch-2 blob format and int4 weights); itofs_sched.c / .h: boot calibration and start-delay policy; BLOB_FORMAT.md: the weight blob format
firmware/    ESP-IDF app: main.c (I2S, serial commands, self-test, board benchmark, weight sets and boot self-calibration, start-delay planning), kernels_s3.c (PIE int8 GEMM via esp-nn, int4 unpack), partitions.csv (three weight partitions)
host/        Makefile, ito_cli.c, host_test_v3.c, gen_selftest.c, opcount_v3.c, w4_dump.c (int4 GEMM test helper), host_test_sched.c (policy test with simulated timings), golden/ (test references)
tools/       fetch_weights.sh, flash.sh, say.py, chip_wav.py, build_fw.sh, int4_check.py, i4_to_i8.py, estimate_v3.py, icount_estimate.py, trace_sim.py, trace_table.py, sched_eval.py, qemu/run_qemu.sh, qemu/icount_profile.sh, qemu/qemu_client.py
prebuilt/    ito_app_merged.bin
```
