# Ito on the ESP32-S3

The whole text-to-speech chain runs on an **ESP32-S3-DevKitC-1 N16R8** (16 MB flash, 8 MB octal PSRAM): text
encoder, duration head, prosody net, mel head, harmonic F0 source, ConvNeXt vocoder and iSTFT. Phonemisation runs on
the host that sends the text (`tools/say.py`). The board receives token ids and plays 24 kHz audio over I2S.

> **Status (2 October 2026).** The engine and firmware are verified on the host and in Espressif's QEMU: the firmware's
> PCM is bit-identical to the host build. **Nothing has run on silicon yet.** Every timing here is an **estimate**
> from counted operations and an assumed GEMM throughput. At boot the firmware measures the real numbers itself and
> prints them (`BOARD_SUMMARY`). We will publish board measurements as soon as we have them.

## 1. Flash and talk

Wiring for an I2S DAC or amplifier:
- BCLK to GPIO15, LRCK to GPIO16, DIN to GPIO17
- PCM5102A: SCK to GND and XSMT high
- or a MAX98357A on 5 V

```bash
pip install esptool pyserial phonemizer espeakng-loader nltk huggingface_hub
huggingface-cli login                                  # once, after accepting the terms at huggingface.co/lokutor-ai/ito-tts-v3
esp32/tools/fetch_weights.sh                           # -> models/ito_v3_esp32s3.bin (flash.sh runs it if needed)
esp32/tools/flash.sh /dev/ttyUSB0                     # prebuilt app at 0x0 + the voice model at 0x200000 (voice D)
VOICE=g esp32/tools/flash.sh /dev/ttyUSB0             # or voice G (male): ito_v3_G_esp32s3.bin at 0x200000
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # press RST and wait for READY (the boot benchmark takes 1-2 min)
python esp32/tools/say.py "Good morning! The coffee is ready." --port /dev/ttyUSB0
```

If flashing fails at 921600 baud, use `BAUD=460800`. If the board does not connect, hold BOOT, tap RST, release BOOT
and try again.

| file | what |
|---|---|
| `prebuilt/ito_app_merged.bin` (362 KB) | bootloader + partition table + app (ESP-IDF 5.5.1, octal PSRAM, QIO flash, I2S on). Flash at 0x0. |
| `ito_v3_esp32s3.bin` (4.89 MB, from [Hugging Face](https://huggingface.co/lokutor-ai/ito-tts-v3), see [`models/README.md`](../models/README.md)) | voice D (female), with a self-test record and three demo sentences. Flash at 0x200000. **CC BY-NC-SA 4.0 + [`models/TERMS.md`](../models/TERMS.md): non-commercial.** |
| `ito_v3_G_esp32s3.bin` (4.89 MB, same place and license) | voice G (male), same format and size. Flash it at 0x200000 **instead of** voice D (`VOICE=g esp32/tools/flash.sh PORT`, or `flash.sh PORT models/ito_v3_G_esp32s3.bin`). The app is the same for both voices. |

At boot the board does four things, in order:
1. It copies the weights to PSRAM and runs the **self-test**: it synthesizes a golden sentence and compares the hash of
   its PCM with the host C engine's. It must print `PASS`.
2. It runs the **board benchmark** (§4) and keeps the fastest weight-staging mode.
3. It speaks three demo sentences.
4. It prints `READY`.

Serial commands (115200 baud): `say <ids>`, `demo [k]`, `test`, `bench`, `act 8|16`, `wmode 0|1|2`, `stats`, `help`.
The BOOT button repeats the demos.

**Rebuild the app** with ESP-IDF 5.5: `esp32/tools/build_fw.sh` writes `prebuilt/ito_app_merged.bin`. The app contains no
model weights (it is GPLv3 code only); the voice model is the separate, non-commercial weights file.

## 2. Try it without a board

```bash
cd esp32/host && make && make test           # host build of the engine; tests against golden references
python3 esp32/tools/chip_wav.py "Any English text." out.wav    # text -> WAV with the chip's exact arithmetic
esp32/tools/build_fw.sh && esp32/tools/qemu/run_qemu.sh       # the real firmware in Espressif QEMU (>= 9.2.2)
```

`make test` checks the C engine against an engine-numerics PyTorch reference stored in `host/golden/` (voice G:
`make test VOICE=g`, references in `host/golden/g/`; `VOICE=g esp32/tools/qemu/run_qemu.sh` runs voice G in QEMU). It checks
durations, the SNR of every stage, and that streaming output is bit-identical to whole-utterance output. It runs for
8- and 16-bit activations, and again under ASan/UBSan. `ito_cli` prints the work done before the first audio chunk
and per second of audio.

## 3. Memory and compute

| | |
|---|---|
| Weights | 4.89 MB blob: 3.42 M int8 + 0.54 M int16 + 0.08 M f32 parameters (the style FiLM is precomputed into a table) |
| Flash | app 296 KB in a 2 MB slot at 0x10000; weights in a 14 MB slot at 0x200000 |
| PSRAM | **peak 5.9 of 8 MB** (weights copied to PSRAM + stage ring buffers + text buffers for 400 tokens; QEMU) |
| Internal SRAM | **peak 327 of 383 KB** (hot scratch + 4 weight-staging tiles + stacks; QEMU) |
| Work before the first 100 ms of audio | **83.6 M int8 MACs, for any sentence length** (forward GRU, fixed style: the text side runs incrementally) |
| Work per second of audio | 346 M int8 MACs + 1.2 M f32 MACs + ~56 M cycles of other float work (LayerNorm, GELU, FFT, source; 0.23 of a core) |
| Weight traffic | 39 MB of weights read from PSRAM per second of audio (8-frame batches) |

## 4. Estimated time to first audio and real-time factor

**These are estimates, not measurements.** The model: time = int8 MACs / G + other float cycles / 240 MHz. G is the
*effective* int8 GEMM throughput on the board, including PSRAM effects. The float cycle costs are assumptions
(`host/opcount_v3.c`). Full table: [`results/chip/estimates.md`](../results/chip/estimates.md).

| | G = 0.3 GOPS | 0.5 | 1.0 | 2.0 | G needed |
|---|---|---|---|---|---|
| Time to first audio (first 100 ms chunk ready) | 323 ms | 212 ms | 128 ms | 86 ms | **0.54 GOPS for < 200 ms** |
| Real-time factor | 1.39 | 0.92 | 0.58 | 0.40 | **0.45 GOPS for real time** |

Then I2S adds at most one 20 ms DMA buffer.

What we do not know yet is G. The S3's vector unit peaks far above 0.5 GOPS on SRAM-resident data, but the weights
stream from PSRAM. At 8.8 MACs per weight byte, G ≤ 8.8 × (effective PSRAM MB/s), so 40 MB/s gives 0.35 and 80 MB/s
gives 0.70. The `gdma` weight-staging mode overlaps the copy with compute. Only the board can say which mode wins.

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
| **quantised (engine numerics) vs float PyTorch, 20 held-out sentences** | **PESQ-wb 4.52** (median 4.56; 3-s segments 4.56, min 4.47); durations equal on 20/20; log-mel L1 0.080. For scale: float against float with a different noise draw scores 3.82 |
| C engine vs reference, stage by stage | durations exact; text side 90–91 dB, durations 96–98 dB, F0 95 dB; mel head 57–66 dB (8-bit) / 87 dB (16-bit); vocoder spectrum 45–47 dB (8-bit) / 81 dB (16-bit) |
| why the 8-bit SNRs are lower | the reference against **itself** with 1e-7 relative noise gives the same range: an 8-bit quantiser flips a whole step when its input crosses a rounding boundary. The C engine sits inside the reference's own rounding noise |
| C engine audio, PESQ | against float 4.50–4.60, against the reference 4.59–4.62 |
| streaming == whole utterance | **bit-identical** for chunks of 1, 2, 3, 5, 8, 16 and 32 frames; incremental and whole-sentence text side bit-identical |
| ASan / UBSan | clean |
| **QEMU** (Espressif 9.2.2, quad PSRAM, I2S compiled out) | self-test PASS; full PCM 134100/134100 samples **bit-identical** to the host; holds on 1 and 2 cores, 8- and 16-bit activations, direct and copy weight staging ([`results/chip/qemu_v3.log`](../results/chip/qemu_v3.log)) |

Timings printed under QEMU are emulator wall-clock times and say nothing about the chip.

## 6. Not verified yet

- **Anything on silicon:** all timings, the octal PSRAM bandwidth, the dual-core speed-up, GDMA staging (QEMU has no
  GDMA, so it reports "unavailable"), I2S output, the BOOT button, and `say.py` over real USB.
- **The hardware build itself** (octal PSRAM, QIO, I2S) was built but never run. QEMU runs the quad-PSRAM, no-I2S build
  of the same sources.
- The float-work cycle costs behind the estimates are assumptions.
- No formal listening test compares the quantised chip output with the float model. The PESQ of 4.52 against float
  suggests the difference is small.

## Layout

```
engine/      itofs.c / itofs.h: portable C99 engine (also reads the older arch-2 blob format)
firmware/    ESP-IDF app: main.c (I2S, serial commands, self-test, board benchmark), kernels_s3.c (PIE int8 GEMM via esp-nn)
host/        Makefile, ito_cli.c, host_test_v3.c, gen_selftest.c, opcount_v3.c, golden/ (test references)
tools/       fetch_weights.sh, flash.sh, say.py, chip_wav.py, build_fw.sh, estimate_v3.py, qemu/run_qemu.sh, qemu/qemu_client.py
prebuilt/    ito_app_merged.bin
```
