# Ito on the ESP32-S3

The whole text-to-speech chain runs on an **ESP32-S3-DevKitC-1 N16R8** (16 MB flash, 8 MB octal PSRAM): text
encoder, duration head, prosody net, mel head, harmonic F0 source, ConvNeXt vocoder and iSTFT. Phonemisation runs on
the host that sends the text (`tools/say.py`). The board receives token ids and plays 24 kHz audio over I2S.

> **Status (5 October 2026).** The engine and firmware are verified on the host and in Espressif's QEMU: the firmware's
> PCM is bit-identical to the host build. **Nothing has run on silicon yet.** Every timing here is an **estimate**
> from exact QEMU instruction counts and an assumed CPI and PSRAM bandwidth. **Real-time playback is estimated at 0.63-0.66 centrally (0.65 for long sentences) and 0.97-0.99 pessimistically, but not established** (§4). At boot the firmware measures the real numbers itself and
> prints them (`BOARD_SUMMARY`). We will publish board measurements as soon as we have them.

## 1. Flash and talk

Wiring for an I2S DAC or amplifier:
- BCLK to GPIO15, LRCK to GPIO16, DIN to GPIO17
- PCM5102A: SCK to GND and XSMT high
- or a MAX98357A on 5 V

```bash
pip install esptool pyserial phonemizer espeakng-loader nltk huggingface_hub
huggingface-cli login                                  # once, after accepting the terms at huggingface.co/lokutor-ai/ito
esp32/tools/fetch_weights.sh                           # -> models/ito_female_esp32s3.bin (flash.sh runs it if needed)
esp32/tools/flash.sh /dev/ttyUSB0                     # prebuilt app at 0x0 + the voice model at 0x200000 (female voice)
VOICE=male esp32/tools/flash.sh /dev/ttyUSB0           # or the male voice: ito_male_esp32s3.bin at 0x200000
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # press RST and wait for READY (the boot benchmark takes 1-2 min)
python esp32/tools/say.py "Good morning! The coffee is ready." --port /dev/ttyUSB0
```

If flashing fails at 921600 baud, use `BAUD=460800`. If the board does not connect, hold BOOT, tap RST, release BOOT
and try again.

| file | what |
|---|---|
| `prebuilt/ito_app_merged.bin` (420 KB) | bootloader + partition table + app (ESP-IDF 5.5.1, octal PSRAM, QIO flash, I2S on). Flash at 0x0. |
| `ito_female_esp32s3.bin` (3.81 MB, from [Hugging Face](https://huggingface.co/lokutor-ai/ito), see [`models/README.md`](../models/README.md)) | female voice, with a self-test record and three demo sentences. Flash at 0x200000. **CC BY-NC-SA 4.0 + [`models/TERMS.md`](../models/TERMS.md): non-commercial.** |
| `ito_male_esp32s3.bin` (3.81 MB, same place and license) | male voice, same format and size. Flash it at 0x200000 **instead of** the female voice (`VOICE=male esp32/tools/flash.sh PORT`, or `flash.sh PORT models/ito_male_esp32s3.bin`). The app is the same for both voices. |

At boot the board does four things, in order:
1. It copies the weights to PSRAM and runs the **self-test**: it synthesizes a golden sentence and compares the hash of
   its PCM with the host C engine's. It must print `PASS`.
2. It runs the **board benchmark** (§4) and keeps the fastest weight-staging mode.
3. It speaks three demo sentences.
4. It prints `READY`.

Serial commands (115200 baud): `say <ids>`, `demo [k]`, `test`, `bench`, `act 8|16`, `wmode 0|1|2|3` (direct, copy, gdma, gdma + prefetch),
`first <frames>` (frames of the first chunk), `delay <ms>` (hold playback until this long after the text arrived), `unpack pie|c` (int4 weight rows: PIE kernel or C code, same bytes), `stats`, `help`.
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
| Flash | app 340 KB in a 2 MB slot at 0x10000; weights in a 14 MB slot at 0x200000 |
| PSRAM | **peak 5.3 of 8 MB** (weights copied to PSRAM + stage ring buffers + PCM chunk buffers + text buffers for 400 tokens; QEMU; 6.5 MB with the 256-wide vocoder) |
| Internal SRAM | **peak 314 of 384 KB in QEMU, about 327 KB on the chip with the I2S DMA buffers** (192 KB hot scratch for 24-frame chunks, 12-row wide layers + 4 weight-staging tiles of 4 KB + stacks; was 340 of 384 KB with a 238 KB scratch for the 256-wide vocoder). The firmware checks at boot that the scratch fits in one block with room left over and falls back to PSRAM (slow) if not: look for `internal SRAM before the hot arena` in the boot log |
| Work before the first audio | **23.0-23.5 M instructions and 3.9 MB of weights from PSRAM for the shipped 10-frame (125 ms) first chunk; 16.0-16.1 M and 3.45 MB for a 2-frame one (`first 2`). The same for any sentence length** (forward GRU, fixed style: the text side runs incrementally) |
| Work per second of audio | 263 M int8 MACs + 1.2 M f32 MACs + other float work; about 85 M instructions on the dual-core critical path |
| Weight traffic | 11 MB of weights read from PSRAM per second of audio (one weight pass serves a 24-frame, 300 ms chunk, the two 1202-wide layers two passes; 15 MB with the 256-wide vocoder, 17 MB before that, 38 MB with 8-frame chunks) |

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
(the pessimistic column of §4: whole-sentence RTF 0.92-0.93 against 0.97-0.98). Where it is limited by the CPU the int4 set is a hair slower than int8 (optimistic 0.47 against 0.46).

## 4. Estimated time to first audio and real-time factor

**Estimated from exact QEMU instruction counts. Not measured on silicon.** QEMU runs the real firmware with `-icount shift=0`
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
| **this engine, 192-wide vocoder, same ramp** | **23.0-23.5 M instructions, 3.9 MB** | **about 85 M instructions, 11 MB** (+ about 4.5 MB of activation rings) |

The counts are the same for short (25 tokens), median (108) and long (175 tokens) sentences. The output is bit-identical between the engines and to the host build.

| Estimated (not measured) | optimistic | central | pessimistic |
|---|---|---|---|
| Time to first audio, first release engine | 338 ms | 463 ms | 672 ms |
| Time to first audio, previous engine (8-frame chunks) | 124-127 ms | 176-179 ms | 266-269 ms |
| Time to first audio, version of 2 October (25 ms first chunk, **gap after it**) | 94-97 ms | 145-148 ms | 237-240 ms |
| Time to first audio, this engine with `first 2` (same idea, **gap after it**) | 87 ms | 125-126 ms | 193-194 ms |
| Time to first audio, engine of 3 October (256-wide vocoder), shipped ramp | 137-143 ms | 200-207 ms | 311-318 ms |
| **Time to first audio, this engine (192-wide vocoder), shipped start-up ramp (no gap after it)** | **124-127 ms** | **171-175 ms** | **251-255 ms** |
| Real-time factor, previous engine | 0.72-0.78 | 1.15-1.23 | 1.93-2.04 |
| Real-time factor, version of 2 October (long-sentence limit) | 0.53-0.55 | 0.77-0.79 | 1.18-1.22 |
| Real-time factor, engine of 3 October (256-wide vocoder; long-sentence limit) | 0.53 | 0.76 | 1.14 |
| **Real-time factor, this engine** (long-sentence limit) | **0.47** | **0.65** | **0.96** |
| Real-time factor, this engine, whole sentences including the start-up ramp | 0.43-0.47 | 0.63-0.66 | 0.97-0.99 |
| Start delay for gapless playback, version of 2 October | about 220 ms | 342-346 ms | 850-2700 ms |
| Start delay for gapless playback, engine of 3 October (256-wide), shipped ramp | 137-143 ms | 215-240 ms | 880-2200 ms |
| **Start delay for gapless playback, this engine, shipped ramp** | **124-127 ms (= time to first audio)** | **176-179 ms** | **560-660 ms** (short sentences need the most; this column is a hair under RTF 1 and is not a guarantee) |

(ranges: the self-test sentence and the three demo sentences, 25 / 84 / 108 / 175 tokens, both voices. The 192-wide figures were recomputed from fresh QEMU instruction counts of the final blobs of both voices.)

Then I2S adds at most one 20 ms DMA buffer.

Reading this honestly:
- **Time to first audio is not the same as gapless speech, so the engine now starts with a ramp.** Every chunk costs one pass over the weights plus about
  6.5 ms (central) per frame; one frame is 12.5 ms of audio. A first chunk of 2 frames is out after about 125 ms, but it holds only 25 ms of audio and the
  24-frame chunk behind it needs about 180 ms more: the speech has a gap of about 150 ms (the version of 2 October needed a 340 ms start delay
  against it). The shipped schedule instead makes the first chunk 10 frames (125 ms of audio), then 11, 12, 14, 18 and 24, with the text side advancing in 8-token
  steps meanwhile (one 24-token step would make a single chunk 65 ms slower than its neighbours): each chunk is then ready before the audio of
  the chunks before it has played out, so playback can start with the first chunk. The cost is a later first sound (171-175 ms central instead of 125-126). In the
  central case the speech is gapless from 176-179 ms after the text arrived (the first chunk plus 3-5 ms, spent on the chunks that carry a text step);
  `delay <ms>` holds playback for that long. In the optimistic case the first chunk is already gapless. In the pessimistic case the whole-sentence RTF is 0.97-0.99 for
  sentences of 84-175 tokens and 1.05 for the 25-token one (the ramp is a larger share of a short sentence), so the start delay is 560-660 ms and a very slow board would still have gaps.
  `first 2` restores the old lowest-latency start (first sound at 125-126 ms central, with the gap). The audio is bit-identical for every schedule (host test G, QEMU).
- **The pessimistic case is only just under real time.** At 40 MB/s of PSRAM bandwidth with no overlap and CPI 1.6 the estimate is RTF 0.96 (long sentences; it was 1.14 with the 256-wide vocoder),
  and 0.97-0.99 for whole sentences with the ramp. That is not a margin. The CPU alone is 0.46 / 0.52 / 0.57 of real time (optimistic / central / pessimistic); the PSRAM bandwidth assumption decides the rest. This column is defined as
  "nothing overlapped", so no change to the staging can improve it; only fewer weight bytes or fewer instructions can.
- **What the central column assumes, and a structural check.** The central estimate hides half of the PSRAM time behind compute. To see what the firmware's
  staging can actually do, `tools/trace_sim.py` replays the exact per-call trace of a chunk (every GEMM call and every row-parallel float loop, with its exact
  instruction count per core and weight bytes) through a model of the weight DMA: both cores share one PSRAM bus, each has two staging buffers. Still an estimate, but without the hidden-share knob:

  | long-sentence RTF (24-frame chunks) | optimistic (CPI 1.3, 80 MB/s) | central (1.45, 60) | pessimistic (1.6, 40) |
  |---|---|---|---|
  | nothing overlapped | 0.66 | 0.78 | 0.96 |
  | GDMA double buffer inside each GEMM call (`wmode 2`, the staging of the version of 2 October) | 0.59 | 0.69 | 0.87 |
  | **plus cross-call prefetch (`wmode 3`)**: the engine announces the next call's weights, which are fetched during the float work in between | **0.55** | **0.64** | **0.78** |

  (Recomputed for the 192-wide vocoder with `tools/trace_table.py` on the 175-token sentence; the same script on the 3 October log reproduces the earlier 0.78 / 0.92 / 1.14, 0.67 / 0.80 / 1.01, 0.63 / 0.74 / 0.91.)

  So the central column's assumption (half hidden) is about what `wmode 3` reaches (RTF 0.64 against the column's 0.65); `wmode 2` hides less (RTF 0.69). With the prefetch the gapless
  start delay is 167-170 ms central and the time to first audio the same (same model; 143-146 ms for both, optimistic). The same model says the bus, not the CPU, limits the GEMM calls (the two cores together ask
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
- **Real time is still not established on silicon.** Only the board can say. If the effective PSRAM bandwidth is 60 MB/s or more, real time holds with margin.
  If not, the fallbacks are int4 weights in the ConvNeXt blocks and fewer blocks (a 4-block fine-tune exists and is being added as a fallback weight set).

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

## 4b. Boot calibration and start-delay policy

`engine/itofs_sched.c` is the policy that turns **measured** chunk times into decisions. It has no clock of its own: the firmware feeds it microseconds, the tests feed it simulated ones.
- *Measured RTF of a trace* (one representative sentence synthesised chunk by chunk with the shipped schedule): the larger of the median full-chunk figure and the whole-trace figure (ramp included).
- *Choice among weight sets* (best quality first): the first whose measured RTF is <= 0.85; otherwise the lowest, reported as marginal (0.85 to 0.95) or **degraded** (>= 0.95).
- *Start delay per utterance* (an adaptive jitter pre-buffer): a model of the speaker and of the 6 PCM buffers, driven by the measured chunk times times a 1.25 safety factor (a trace's own chunks, then its steady median, and a fixed-plus-per-frame fit for the short last chunk),
  returns the smallest delay with no predicted underrun and never less than the first chunk's production time; the chunk times actually seen in the utterance replace the model's as they arrive.

`make test_sched` (host) checks it with simulated timings against an independent discrete-event model of the speaker (true chunk times of a *different* sentence with its own jitter): the weight-set choice at the exact thresholds and with missing sets;
**8,640 random sentences** (RTF 0.30 to 0.85, three splits of fixed and per-frame cost, 3 to 400 tokens, +-4 % jitter per chunk and +-4 % sentence-to-sentence cost between calibration and playback) with **0 underruns**; the first audio released with no extra wait on a fast board;
a board slower than real time reported infeasible rather than hidden; slower-than-calibrated chunks seen early raising the delay. Outside the margin (+-8 % and +-8 %) 4 of 2,880 sentences underrun, which the test reports as information, not a guarantee.
Nothing here runs on a board yet; the firmware integration is the next section of this README.

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

Timings printed under QEMU are emulator wall-clock times and say nothing about the chip.

## 6. Not verified yet

- **Anything on silicon:** all timings, the octal PSRAM bandwidth, the dual-core speed-up, GDMA staging (QEMU has no
  GDMA: the control flow of `wmode 2` and `wmode 3` is tested there with a software copy, but the real `esp_async_memcpy` branch, its interrupts and the
  two cores sharing the bus are not), I2S output, the BOOT button, and `say.py` over real USB. The boot benchmark runs all four staging modes,
  compares each one's audio with the direct mode's and keeps the fastest that matches.
- **The hardware build itself** (octal PSRAM, QIO, I2S) was built but never run. QEMU runs the quad-PSRAM, no-I2S build
  of the same sources.
- The float-work cycle costs behind the estimates are assumptions.
- No formal listening test compares the quantised chip output with the float model. The PESQ of 4.52 against float
  suggests the difference is small.

## Layout

```
engine/      itofs.c / itofs.h: portable C99 engine (also reads the older arch-2 blob format and int4 weights); BLOB_FORMAT.md: the weight blob format
firmware/    ESP-IDF app: main.c (I2S, serial commands, self-test, board benchmark), kernels_s3.c (PIE int8 GEMM via esp-nn)
host/        Makefile, ito_cli.c, host_test_v3.c, gen_selftest.c, opcount_v3.c, w4_dump.c (int4 GEMM test helper), golden/ (test references)
tools/       fetch_weights.sh, flash.sh, say.py, chip_wav.py, build_fw.sh, int4_check.py, i4_to_i8.py, estimate_v3.py, icount_estimate.py, trace_sim.py, trace_table.py, sched_eval.py, qemu/run_qemu.sh, qemu/icount_profile.sh, qemu/qemu_client.py
prebuilt/    ito_app_merged.bin
```
