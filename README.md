# Ito: natural-sounding streaming text-to-speech for a $5 chip

[![Ito demo: click to watch the video with sound](docs/demo.gif)](https://lokutor-ai.github.io/ito/)

*Click the animation to watch the 50-second video **with sound** (the GIF is silent). Every voice in it is Ito's own output from the chip engine.*

Ito is English text-to-speech that runs entirely on an **ESP32-S3** (240 MHz dual-core Xtensa LX7, 8 MB PSRAM, 16 MB
flash), with no cloud and no neural accelerator. It streams: audio starts after a short first chunk (125 ms) is ready, and the work
before it does not grow with the sentence length. The voice is distilled from a large open TTS model into 3.3 M parameters. Built by
[Lokutor](https://lokutor.com), the makers of [Oído](https://github.com/lokutor-ai/oido) (speech recognition on the
same chip).

> **First run on a physical board (8 October 2026):** the first public prebuilt image crash-looped on first boot on real hardware: its GDMA self-test overflowed a heap buffer, which QEMU cannot show. That run found four first-silicon bugs (the GDMA self-test heap overflow, bench scratch fragmentation, underrun accounting, USB console input), fixed in `fedbd26` (with the notes in `e196730`). The prebuilt image in this repository is the fixed one, and the logs of the crashing runs are kept in `esp32/logs/board_crash_*.log`. The measured numbers below come from the fixed image.
>
> **Status (9 October 2026).** The on-chip engine is verified on a laptop and in Espressif's QEMU emulator: the firmware's audio is bit-identical to the host build of the engine, and every sample in [`samples/`](samples) is that engine's exact output. On the physical board the same self-test also passed: the chip's audio is bit-identical to the host C engine (134100 samples).
>
> **Measured on a board** (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice): with the main int8 set the first audio chunk (125 ms of audio) was computed and ready 184 ms after the text was handed to the engine (183.9–184.3 ms over the three demo sentences), and the real-time factor was 0.66 on the boot calibration sentence (0.660–0.674 on the demo sentences). Gap-free playback is a separate number: the firmware plans to start playback about 246 ms after the text arrives, and with that planned delay the three demo sentences had 0 underruns. No DAC or amplifier was attached and no audio was listened to, so the I2S/DAC output stage is not included, these are not acoustic measurements, and the underrun count is the firmware's own accounting against its playout clock. The first audio chunk is ready in under 200 ms and synthesis runs faster than real time on this board (RTF 0.66), with 0 underruns when playback starts after the planned 246 ms delay. For the female voice, the other two weight sets were timed by hand once on 8 October with the `tier` serial command (one run each, not chosen by the boot calibration): main int4 set RTF about 0.65, first chunk about 180 ms; light set RTF about 0.61, first chunk about 168 ms. **Update, 9 October 2026** (same board, same firmware fedbd26, still no DAC and nothing listened to). Male voice: the boot calibration chose the main int8 set (RTF 0.659 on the calibration sentence, in both of two boots), the bit-exact self-test passed (137,400 samples, identical to the host C engine), the first audio chunk was ready in 183.8–184.8 ms over the demo sentences, the demo-sentence RTF was 0.659–0.672, the firmware planned a gap-free start 248–249 ms after the text arrives, and underruns were 0. The male voice's other two sets were timed by hand with `tier` (not boot-chosen; the firmware plans no start delay for hand-timed sets), demo sentences only: main int4 set first chunk 179.5–179.8 ms, RTF 0.645–0.653 (two passes); light set first chunk 167.9–168.0 ms, RTF 0.606–0.613 (one pass); the self-test passed with each set. Female voice, repeats: the same image booted four times in all (8 October and three times on 9 October, each after a USB-serial or watchdog reset, not a cold power cycle); the first chunk was ready in 183.5–184.5 ms over all demo runs, the boot-calibration RTF was 0.660–0.661, the planned start was 246–248 ms, the bit-exact self-test passed every time and underruns were 0. After a demo the PSRAM peak was 5372 KB (female) and 5380 KB (male) of 8192 KB, and internal SRAM 354 of 374 KB (stack high-water marks were not printed). All of this comes from one board: the repeats and the male runs show repeatability on one chip, not a second board. Against the pre-board estimates (optimistic / central / pessimistic: RTF 0.43–0.47 / 0.63–0.66 / 0.97–0.99, first audio chunk 124–132 / 171–180 / 251–260 ms), the measured RTF of 0.66 is at the top of the central estimate and far from the pessimistic one (0.97–0.99), and the measured first chunk of 184 ms is slightly above the central estimate and below the pessimistic one. Still not measured: audio through a DAC and any listening to the output (no DAC was attached), a second board or a second chip revision, cold power-on boots (every boot was a reset), power draw, long-run thermal behaviour and supply sensitivity, stack high-water marks (not printed), the male voice's sound quality through the chip (its audio is bit-identical to the host engine: self-test PASS), and a planned start delay for the hand-timed sets (the firmware plans none). Still estimates: the optimistic, central and pessimistic columns, which are the pre-board predictions kept for comparison. In the boot benchmark the PSRAM-to-SRAM copy ran at 86.0 MB/s (above the 40–80 MB/s the estimates assumed), GDMA at 36.4 MB/s (slower than a plain copy), cached PSRAM reads at 89.2 MB/s. The firmware therefore stages weights with `direct` reads, the fastest of the four modes it benchmarks (mean RTF 0.66, GEMM 1.060 GMAC/s). The vocoder is 192 wide instead of 256; in a blind test (#10, one listener) it sounded the same as the 256-wide one. The engine starts with a 125 ms first chunk and a short ramp so that playback can start with the first chunk without a gap. An earlier version of this page claimed 130–210 ms and real time at 0.5–1 GOPS; that was too optimistic (see [`esp32/README.md`](esp32/README.md) §4a and §4).
>
> **The firmware checks real time by measurement.** Each voice ships three weight sets (main int8; main with int4 blocks; a light set with one block fewer): at boot the board measures the real time of every chunk with each set, keeps the first whose measured RTF is at most 0.85, plans the playback start delay from those measurements, and prints `WARNING: DEGRADED` (and sets a flag) if even the fastest set measures 0.95 or more. On this board the boot calibration chose the main int8 set with a measured RTF of 0.66 (status OK, degraded flag 0; main int4 and light were not needed, so the boot did not time them; they were timed by hand afterwards). The male voice's boot calibration also chose the main int8 set (RTF 0.659, both male boots), and so did all four female boots. This is a check by measurement on the board that runs it, not a promise about other boards; the full boot log of the 8 October run is in the repository as `esp32/logs/board_fixed_boot_and_tiers_2026-10-08.log`, and the other logs, with a note on which is which, are in [`esp32/logs/`](esp32/logs/README.md).

> **Stability on the board (9 October 2026).** Same board and firmware (`fedbd26`), 240 MHz, female voice, USB powered, nothing attached, I2S idle. **Soak:** each iteration runs the boot golden sentence in three modes (dual-core 8-bit, single-core, dual-core 16-bit) and compares the output hash with the host C engine. Main int8 set: 300 iterations (900 runs, all PASS, 0 FAIL, 78 minutes; hashes `023bd6b1` and `9666bb9e`, identical to the host C engine). Int4 set: 100 iterations (300 PASS, 0 FAIL; host hash `363d7d98`). Light set: 100 iterations (300 PASS, 0 FAIL; `5f12d5e1`). Main int8 set again with I2S idle: 100 iterations (300 PASS, 0 FAIL). In all, 600 iterations (1,800 runs) in about 2.5 hours, with no panic, watchdog or reboot in any phase. A 175-token demo repeated 15 times in the main phase gave a first chunk of 183.9–184.6 ms and an RTF of 0.662–0.663, with no drift and no sign of throttling. **Synthetic stress:** Ito's own PIE kernel on both cores in lock-step at 240 MHz for about 21 minutes (3 rounds, random int8 data, every call compared with a one-core reference): K=704 (64 rows, 45 KB shared tile, 88/88 split, the pattern reported by the Oído project), K=576, K=192, K=1216 and a one-core control gave 0 mismatches and 0 crashes. **What this says, and no more:** no corruption or crash was observed in 600 golden-sentence iterations and a 21-minute synthetic stress on one board. **The Oído finding.** The [Oído](https://github.com/lokutor-ai/oido) project reports (["Dual core"](https://github.com/lokutor-ai/oido#dual-core)) that on silicon its two-core mode gives wrong output or crashes when both cores run PIE kernels at the same time; the case described to us was a two-core 240 MHz int8 K=704 streaming kernel that was exact at 160 MHz. Ito did not reproduce that with its own kernel on this board. That is board- and pattern-specific information, not proof that the problem is absent elsewhere: our stress started core 1 by spinning on a flag and the engine starts it with a FreeRTOS notification, a start with both cores aligned to the same cycle was not tested, Ito's real tiles are 5–15 KB, no real layer has K=704, and about 60 % of the time goes to float work between the GEMM calls (see [`esp32/logs/dualcore_load_comparison.md`](esp32/logs/dualcore_load_comparison.md)). Anyone running dual-core PIE streaming at 240 MHz should test it, and on more than one board. **Not covered:** a second board or chip revision, cold power-on boots, supply measurements (the USB voltage was not measured), power draw, temperature, the male voice (not soaked), audio output (no DAC) and long-term (days) stability. Logs: [`esp32/logs/`](esp32/logs/README.md); stress source: [`esp32/tools/stress/`](esp32/tools/stress/README.md).
>
> **Licensing in one line.** The code is GPLv3; the voice model that makes it talk is **non-commercial**
> (CC BY-NC-SA 4.0 + [terms](models/TERMS.md)). Hobby, research and education use is welcome. For anything commercial,
> or if you are a small team that wants to build with it, [talk to us](#license): we like collaborating.

**Hear it:** [samples next to sanoTTS and the teacher, with a blind mode](https://lokutor-ai.github.io/ito/) ·
**Watch:** [the 50 s video, with sound](https://lokutor-ai.github.io/ito/) ·
**Weights:** [lokutor-ai/ito](https://huggingface.co/lokutor-ai/ito)

[![Ito: natural speech from a $5 chip](https://lokutor-ai.github.io/ito/og.png)](https://lokutor-ai.github.io/ito/)

## Listen

Ito's chip output for eight sentences it never saw in training, rendered by the host build of the on-chip engine with
the chip's exact arithmetic:

| | sentence |
|---|---|
| [01.wav](samples/01.wav) | Hey, are you still coming over for dinner tonight, or should I save you a plate? |
| [02.wav](samples/02.wav) | Your package should arrive on Friday, October 9th, sometime before noon. |
| [03.wav](samples/03.wav) | When I finally got to the station, the last train had already left, so I ended up sharing a taxi with two strangers who turned out to be surprisingly good company. |
| [04.wav](samples/04.wav) | The pharmacist recommended an anti-inflammatory, but honestly, I'd rather try physiotherapy first. |
| [05.wav](samples/05.wav) | Thanks so much for calling. I'll check the schedule and get back to you first thing tomorrow morning. |
| [06.wav](samples/06.wav) | Could you grab some quinoa and Worcestershire sauce on your way home? |
| [07.wav](samples/07.wav) | It's about 23 degrees outside, so you probably won't need a jacket. |
| [08.wav](samples/08.wav) | I know it sounds strange, but I actually enjoy the quiet hours before everyone else wakes up. |

The float-model clips used in listening test #9 (sentences 01, 03, 04 and 06 were rated), next to the teacher's, are in
[`results/blind9/audio/`](results/blind9/audio). The clips above are from the 192-wide vocoder that ships now (the earlier release had a 256-wide one;
the demo video on the project page was made with that one).

## Listening tests

**Blind test #9** (2 October 2026). There were 24 clips in random order with the system names hidden, rated 1–5 for
naturalness. Each system read the same four new conversational sentences (a question, a long sentence, hard words, and
a question with hard words).

| System | Parameters | Runs on | Mean score (4 clips) |
|---|---|---|---|
| Teacher: StyleTTS 2 (LibriTTS model, reference voice) | large (diffusion, PL-BERT) | GPU / laptop | 4.75 |
| **Ito**, streaming front (as rated: 256-wide vocoder; the shipped 192-wide one is 3.3 M, see #10) | **4.4 M** | **ESP32-S3** | **4.00** (4/4/4/4) |
| Ito variant with a bidirectional front (no streaming text side) | 4.5 M | ESP32-S3 | 4.00 |
| Ito variant with a wider vocoder | 8.2 M | over the chip budget | 3.75 |
| sanoTTS "amy" | 1.45 M | — | 2.00 |
| sanoTTS "heart-nano" | 0.29 M | — | 1.00 |

Please read these numbers with their limits:
- **There was one listener**, an expert (Lokutor's founder), and **four sentences per system**. This is a strong
  direction, not a statistically powered MOS study. With n = 4, differences under about half a point are noise.
- **The rated Ito clips are the float PyTorch model**, through the same streaming path as the chip. They used a small
  optional text-to-style predictor. The shipped chip uses a fixed mean style instead, so that the time to first audio
  does not grow with sentence length. In a paired comparison on held-out sentences, the predictor changed the log-mel
  distance to the reference by only −0.009. The quantised engine scores PESQ 4.5 against the float model. Test #10 (below) rated the chip engine itself (female voice).
- All clips were trimmed and loudness-normalised to −20 LUFS at 24 kHz. Ratings, key, sentences and automatic metrics
  are in [`results/blind9/`](results/blind9).

**Blind test #10** (5 October 2026), the lighter vocoder. Female voice, the same four sentences (01, 03, 04, 06), 16 clips in random
order with the system names hidden, one listener (the same expert), 1–5 naturalness:

| System | Mean score (4 clips) |
|---|---|
| Teacher (reference) | 4.25 |
| Ito with the 256-wide vocoder (the chip engine, int8; shipped until 5 October) | 4.00 |
| **Ito with the 192-wide vocoder (the chip engine, int8; shipped now)** | **4.00** |
| Ito with the 192-wide vocoder and int4 weights in the ConvNeXt blocks (int4 emulation) | 4.00 |

He heard no difference between the three Ito systems. One listener and four clips per system: this says "no difference he could hear", not that there is none.
Automatic checks on 60 held-out utterances agree (mel distance +0.014 / +0.002 for female / male, UTMOS within 0.02). Summary in [`results/blind10/`](results/blind10).

Earlier, **blind test #8** (one listener, four held-out sentences) compared two earlier Ito variants with the teacher.
The scores were teacher 4.00, variant A 3.25, variant B 3.00. In #6
and #7 the gap to the teacher had been 2.25 to 2.5 points, and in #9 it is 0.75. Details are in
[`results/blind8/`](results/blind8).

**Automatic metrics** on the eight sentences above (UTMOS; Whisper word error rate after number normalisation; Ito measured with the first release's 256-wide vocoder):

| | Teacher | Ito | sanoTTS amy | sanoTTS heart-nano |
|---|---|---|---|---|
| UTMOS | 4.49 | 4.46 | 3.98 | 2.07 |
| WER, Whisper medium.en / base | 0 / 0 | 0 / 0 | 0 / 1.0 % | 1.0 / 1.0 % |

UTMOS cannot hear intonation, so treat it as a check, not a verdict. On 60 held-out sentences, Ito's pitch range is
0.94 of the teacher's (female voice; 0.97 male): the per-utterance spread of log-F0 relative to the teacher.

**Benchmark against other small and embedded TTS systems** (54 prompts, 21 public systems and the teacher):

Automatic metrics (mean over the prompts; Whisper large-v3 word error rate; higher is better except WER). Ito is the chip engine's exact output (host build, main int8 weight set); the 256-wide row is the first release, kept for comparison. All numbers, confidence intervals, paired differences and methods are in [`bench/`](bench).

| System | Params | Runs on a microcontroller | UTMOSv2 | UTMOS22 | DNSMOS | WER % |
|---|---|---|---|---|---|---|
| *Built for or run on microcontrollers* | | | | | | |
| Ito, shipped, female (chip-exact, ESP32-S3 emulated) | 3.34 M (2.99 M on chip) | ESP32-S3 (emulated, bit-exact) | 3.21 | 4.43 | 3.34 | 0.6 |
| Ito, shipped, male | 3.34 M (2.99 M on chip) | ESP32-S3 (emulated, bit-exact) | 3.26 | 4.41 | 3.43 | 0.7 |
| Ito, first release (256-wide vocoder) | 4.05 M | ESP32-S3 (emulated, bit-exact) | 3.15 | 4.44 | 3.39 | 0.4 |
| Inflect Nano v2 (Owen Song) | 3.96 M | ESP32-P4, 3.5x slower than real time | 3.08 | 4.41 | 3.40 | 1.1 |
| TinyTTS | 1.6 M | ESP32-S3, 22.9x slower than real time | 2.45 | 3.66 | 3.29 | 6.8 |
| sanoTTS amy | 1.45 M | no (the MCU voice is a different 567 K one) | 2.80 | 3.96 | 3.18 | 1.2 |
| sanoTTS heart-nano | 0.29 M | MCU-sized, no published timing | 1.33 | 2.17 | 2.97 | 1.7 |
| eSpeak NG (rules) | - | community ports | 1.74 | 2.14 | 2.76 | 0.3 |
| *Larger models (CPU or GPU)* | | | | | | |
| Inflect Micro v2 (Owen Song) | 9.36 M | no | 3.46 | 4.41 | 3.38 | 1.0 |
| Kitten TTS nano | 14.0 M | no | 1.99 | 3.93 | 3.31 | 1.1 |
| Piper amy low | 15.6 M | no | 3.42 | 4.44 | 3.28 | 0.7 |
| Piper lessac medium | 15.7 M | no | 3.69 | 4.28 | 3.29 | 0.7 |
| MeloTTS EN | 51.9 M | no | 3.03 | 3.72 | 3.01 | 3.2 |
| Supertonic 2 | 65.5 M | no | 3.62 | 4.44 | 3.35 | 2.4 |
| Kokoro-82M | 81.8 M | no | 3.87 | 4.49 | 3.41 | 0.8 |
| Supertonic 3 | 99.2 M | no | 3.84 | 4.45 | 3.32 | 1.7 |
| MOSS-TTS-Nano | ~100 M | no | 3.44 | 4.37 | 3.21 | 1.7 |
| Pocket TTS | 110 M | no | 3.31 | 4.33 | 3.31 | 2.5 |
| StyleTTS 2 (Ito's teacher) | 191 M | no | 3.43 | 4.47 | 3.34 | 1.5 |

What this says, and no more: among the neural systems built for or run on a microcontroller, Ito (female) has the highest UTMOSv2 (+0.13 over Inflect Nano v2, 95% interval 0.03 to 0.23), ties it on UTMOS22, is behind it on DNSMOS (3.34 vs 3.40) and has the lower word error rate (0.6 vs 1.1 %). Nine larger models, including the teacher, score higher on UTMOSv2, and none of them runs on an MCU. UTMOSv2 is stochastic (about 0.03 on these 54-clip means), the male and female rows are different speakers, and the Ito rows were scored on a different machine from the other systems' stored values (deterministic metrics reproduce; see [`bench/README.md`](bench/README.md)). Inflect's speed is a third-party measurement; Ito's is an estimate.

**Prior art:** see [`docs/prior_art.md`](docs/prior_art.md).

## Size and compute

| | |
|---|---|
| Parameters | 3.34 M: acoustic front 1.62 M + vocoder 1.72 M (was 4.40 M with a 256-wide vocoder) |
| Chip weights | **3.81 MB** main set (`ito_female_esp32s3.bin`; was 4.89 MB): int8 mel head and vocoder; int16 pitch path. Fallback sets for the boot calibration: main-int4 3.20 MB (`_int4`), light 3.05 MB (`_light`) |
| Flash | 379 KB app + three 4.5 MB weight partitions (main 3.81 MB, main-int4 3.20 MB, light 3.05 MB per voice) |
| PSRAM / SRAM | peak 5.2 of 8 MB PSRAM and 354 of 374 KB internal SRAM, measured on the board after a demo: PSRAM peak 5372 KB female / 5380 KB male of 8192 KB (the pre-board QEMU figures were peak 5.3 of 8 MB PSRAM, 320 of 379 KB internal SRAM) |
| Work before the first audio (125 ms first chunk) | **23.0–24.3 M instructions and 3.9 MB of weights read from PSRAM, for any sentence length** (exact counts from QEMU; 16.0–16.5 M and 3.45 MB for a 25 ms first chunk) |
| Work per second of audio | 76–86 M instructions on the dual-core critical path, 10.5–11.2 MB of weights read from PSRAM (24-frame chunks; exact counts; the conversion to time is an estimate) |
| Time to first audio | **measured: 184 ms** until the first audio chunk (125 ms of audio) is computed and ready after the text is handed to the engine (183.9–184.3 ms over the three demo sentences of that run; main int8 set; excludes the I2S/DAC output stage: no DAC was attached and no audio was listened to, so this is not an acoustic measurement; ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice). Pre-board estimates for this set: 124–132 / 171–180 / 251–260 ms (optimistic / central / pessimistic). Further runs on the same board and firmware, 9 October 2026, main int8 set: male voice 183.8–184.8 ms over the demo sentences (two boots); female voice 183.5–184.5 ms over all demo runs of four boots (8 October and three on 9 October, all after resets, none a cold power cycle); in every case the planned gap-free start was 246–249 ms, not 184 ms. The first chunk is 125 ms of audio and the chunks behind it are sized so that playback can start with it without a gap (the version of 2 October made 25 ms of sound after 145 ms and then fell silent for about 200 ms; `first 2` still does that) |
| Real-time factor | **measured: 0.66** on the boot calibration sentence (main int8 set; demo sentences 0.660–0.674; below 1 is faster than real time; ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice). Pre-board estimates for this set: 0.43–0.47 / 0.63–0.66 / 0.97–0.99 (optimistic / central / pessimistic). Female voice, timed by hand once, not chosen by the boot calibration: main int4 set RTF about 0.65, first chunk about 180 ms; light set RTF about 0.61, first chunk about 168 ms. Further runs, 9 October 2026: female boot-calibration RTF 0.660–0.661 over four boots; male voice boot calibration 0.659 (demo sentences 0.659–0.672); male voice timed by hand, not boot-chosen: main int4 set RTF 0.645–0.653 and first chunk 179.5–179.8 ms (two passes), light set RTF 0.606–0.613 and first chunk 167.9–168.0 ms (one pass) |
| Start delay for gapless speech | **measured: about 246 ms** (246–247 ms over the three demo sentences) is the delay the firmware plans before playback starts so that speech is gap-free, planned from its own calibration; with it the three demo sentences had 0 underruns (the firmware's own accounting; no audio was output or heard; main int8 set; ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 October 2026, firmware fedbd26, female voice). Further runs, 9 October 2026: 246–248 ms over four female boots; male voice 248–249 ms, with 0 underruns (the firmware plans no start delay for hand-timed sets). Pre-board estimates for this set: 125–130 / 176–182 / 558–663 ms (optimistic / central / pessimistic). `delay <ms>` on the serial console holds playback for a chosen time |
| On a laptop | RTF ≈ 0.01 on an Apple M4 Max CPU, for both the PyTorch model (whole utterance) and the C engine |

The first row-group above is a **measurement on one board** (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), firmware fedbd26; female voice on 8 and 9 October 2026, male voice on 9 October 2026); the work-per-chunk figures (instructions, bytes) are exact counts from QEMU. Against the pre-board estimates (optimistic / central / pessimistic: RTF 0.43–0.47 / 0.63–0.66 / 0.97–0.99, first audio chunk 124–132 / 171–180 / 251–260 ms), the measured RTF of 0.66 is at the top of the central estimate and far from the pessimistic one (0.97–0.99), and the measured first chunk of 184 ms is slightly above the central estimate and below the pessimistic one. In the boot benchmark the PSRAM-to-SRAM copy ran at 86.0 MB/s (above the 40–80 MB/s the estimates assumed), GDMA at 36.4 MB/s (slower than a plain copy), cached PSRAM reads at 89.2 MB/s. The firmware therefore stages weights with `direct` reads, the fastest of the four modes it benchmarks (mean RTF 0.66, GEMM 1.060 GMAC/s). No DAC or amplifier was attached and no audio was listened to, so the I2S/DAC output stage is not included, these are not acoustic measurements, and the underrun count is the firmware's own accounting against its playout clock. The male voice and the repeat boots ran on the same single board: they show repeatability on one chip, not a second board, and every boot was a reset, not a cold power cycle.
The estimate model and full tables are in [`esp32/README.md`](esp32/README.md); the optimistic / central / pessimistic columns remain the pre-board estimates and were not changed by the measurements.

**The three weight sets, pre-board estimates** (optimistic / central / pessimistic; whole sentences with the start-up ramp; the last rows are measured on one ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 and 9 October 2026, no DAC attached, nothing listened to):

| | main int8 | main int4 | light (4 blocks, int4) |
|---|---|---|---|
| time to first audio | 124–132 / 171–180 / 251–260 ms | 127–133 / 168–174 / 236–243 ms | 118–124 / 157–164 / 222–229 ms |
| real-time factor | 0.43–0.47 / 0.63–0.66 / 0.97–0.99 | 0.45–0.48 / 0.62–0.64 / 0.92–0.93 | 0.42–0.45 / 0.59–0.61 / 0.88 |
| gapless start delay | 125–130 / 176–182 / 558–663 ms | 127–133 / 173–179 / 481–526 ms | 118–124 / 162–169 / 406–429 ms |
| **measured, female: real-time factor** | **0.66** (chosen; 0.660–0.661 over four boots) | about 0.65 (timed by hand, one run, not boot-chosen) | about 0.61 (timed by hand, one run, not boot-chosen) |
| **measured, female: first audio chunk ready** | **184 ms** (calibration sentence; 183.5–184.5 ms over all demo runs of four boots) | about 180 ms (timed by hand, one run) | about 168 ms (timed by hand, one run) |
| **measured, female: planned start delay for gap-free playback** | **246–247 ms** planned (3 demo sentences, 8 October); 246–248 ms over four boots | not planned for hand-timed sets | not planned for hand-timed sets |
| **measured, male: real-time factor** | **0.659** (chosen at boot, both boots; demo sentences 0.659–0.672) | 0.645–0.653 (timed by hand, two passes, not boot-chosen) | 0.606–0.613 (timed by hand, one pass, not boot-chosen) |
| **measured, male: first audio chunk ready** | **183.8–184.8 ms** (demo sentences, two boots) | 179.5–179.8 ms (timed by hand, two passes) | 167.9–168.0 ms (timed by hand, one pass) |
| **measured, male: planned start delay for gap-free playback** | **248–249 ms** planned (3 demo sentences, both boots) | not planned for hand-timed sets | not planned for hand-timed sets |

int4 weights save a fifth of the weight traffic for about 1 % more instructions (with a PIE unpack kernel), which helps only where the PSRAM is the limit. In the pessimistic column no set reaches the 0.85 target; the boot would keep the light set as "marginal".

## How it works

```
 text ─► espeak-ng phonemes (on the host) ─► token ids ─────────────────────────────────────── over serial to the chip
                                                 │
   ┌──────────────────────── acoustic front (1.62 M) ────────────────────────┐   ┌────────── vocoder (1.72 M) ─────────┐
   │ embedding ─► 3 conv layers ─► forward GRU ─► duration head (FiLM style) │   │ [log-mel | log-F0 | voiced]          │
   │      ─► length regulation (80 frames/s)                                 │   │  ─► 7-tap conv + harmonic F0 source  │
   │      ─► prosody net: log-F0, voicing, energy                            │──►│  ─► 5 ConvNeXt blocks                │──► 24 kHz
   │      ─► mel head: 100-bin log-mel                                       │   │  ─► magnitude + phase ─► iSTFT       │    audio
   └─────────────────────────────────────────────────────────────────────────┘   └──────────────────────────────────────┘
```

- **Distillation.** Ito learned its voice from a large open model, StyleTTS 2, speaking in the voice of a LibriTTS-R
  reader. The small student predicts durations, pitch, energy and a mel spectrogram from phonemes.
- **Separate front and vocoder.** The acoustic front predicts the mel spectrogram. A Vocos-style ConvNeXt + iSTFT
  vocoder with a harmonic F0 source turns it into audio.
- **Streaming by construction.** The GRU runs forward only, so the text side runs incrementally. Everything at frame
  rate is a convolution with a bounded right context: 23 frames in total, including the iSTFT overlap-add. This is
  lookahead over features derived from the text, which is already known, so it adds no audio latency. Each chunk
  (on the chip a 125 ms first chunk, then 11, 12, 14, 18 and 24 frames of 12.5 ms; the Python `stream()` uses 100 ms ones) is computed from ring buffers. The streamed output is bit-identical to whole-utterance output on the chip, and equal to float precision (about 120 dB) in PyTorch.
- **The engine** (`esp32/engine`) is new portable C99 code. It uses int8 and int16 GEMMs on the S3's vector unit
  through esp-nn, per-row activation scales (so streaming stays bit-exact), its own transcendental functions
  built from basic arithmetic (bit-identical on host and chip), and a mixed-radix FFT. It reads weights from PSRAM with tiled
  staging and can use both cores.

Training code and recipe are not public; contact us for research collaborations (contact@lokutor.com).

## Quickstart

**Get access to the voice model (once).** The weights are distributed through the gated Hugging Face repository
[lokutor-ai/ito](https://huggingface.co/lokutor-ai/ito). Open it, accept the
[terms](models/TERMS.md) (free for research, education and personal projects), then log in on your machine:

```bash
pip install -e .                 # also installs huggingface_hub
huggingface-cli login            # or: hf auth login, or export HF_TOKEN=...
```

Everything below downloads the weights it needs on first use. Details and offline use: [`models/README.md`](models/README.md).

**Python (CPU is fine).** Needs Python ≥ 3.9.

```bash
ito-tts "Good morning! The coffee is ready." -o hello.wav              # female voice (default)
ito-tts --voice male "Good morning! The coffee is ready." -o hello_male.wav  # male voice
```

Two voices: **female** (LibriTTS-R speaker 4970; `ito_female.pt` / `ito_female_esp32s3.bin`) and **male** (LibriTTS-R
speaker 5105; `ito_male.pt` / `ito_male_esp32s3.bin`). Each chip file holds one voice.

```python
from ito import Ito
tts = Ito.load()                                   # female voice; downloads ito_female.pt on first use
tts_g = Ito.load(voice="male")                     # male voice: ito_male.pt
wav = tts.synthesize("Could you grab some quinoa on your way home?")   # float32 numpy, 24 kHz
for chunk in tts.stream("A sentence of any length."):                  # 100 ms chunks (the chip starts with a 125 ms one, then larger ones)
    ...
```

`--style predicted` uses the optional text-to-style predictor, as in the blind test (female voice only). `--stream` uses the
chunked path.

**On a laptop, with the chip's exact arithmetic** (no board):

```bash
cd esp32/host && make && make test && cd ../..      # make test fetches ito_female_esp32s3.bin into models/
python esp32/tools/chip_wav.py "Good morning! The coffee is ready." hello_chip.wav
python esp32/tools/chip_wav.py --voice male "Good morning! The coffee is ready." hello_chip_male.wav   # male voice
```

`make test VOICE=male` runs the same host test on the male voice's chip file.

**On a board.** You need an ESP32-S3-DevKitC-1 **N16R8** and an I2S DAC or amplifier: BCLK→GPIO15, LRCK→GPIO16,
DIN→GPIO17 (PCM5102A or MAX98357A).

```bash
pip install esptool pyserial
esp32/tools/fetch_weights.sh                       # the voice's three weight sets (ito_female_esp32s3{,_int4,_light}.bin) from Hugging Face into models/ (flash.sh also does it)
esp32/tools/flash.sh /dev/ttyUSB0                  # prebuilt app + female voice, all three sets (0x200000 / 0x680000 / 0xB00000)
VOICE=male esp32/tools/flash.sh /dev/ttyUSB0       # ... or the male voice
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # press RST: self-test, benchmark, weight-set calibration, 3 demo sentences, READY
python esp32/tools/say.py "Hello from a five dollar chip." --port /dev/ttyUSB0
```

`say.py` turns the text into phonemes on your computer and sends token ids over serial. Please send us the boot log:
it contains the first real speed measurements.

**In the emulator** (ESP-IDF 5.5, Espressif QEMU ≥ 9.2.2): `esp32/tools/build_fw.sh && esp32/tools/qemu/run_qemu.sh`.

## Training

Training code and recipe are not public. Contact us for research collaborations: contact@lokutor.com.

## Repository

```
ito/             Python package: front.py, vocoder.py, layers.py, synth.py (inference + streaming), text.py (G2P), cli.py
eval/            stream_check.py (streaming vs whole-utterance synthesis, on the CPU)
esp32/           engine/ (C99 engine), firmware/ (ESP-IDF app), host/ (host build, tests, ito_cli), tools/ (flash, say,
                 chip_wav, build, QEMU, estimates), prebuilt/ (app image, code only), README.md
models/          README.md (how to get the weights from Hugging Face), LICENSE-WEIGHTS, TERMS.md
samples/         chip-exact output for eight unseen sentences
results/         blind10/, blind9/ and blind8/ (listening tests), chip/ (host tests, QEMU log, op profile, estimates)
bench/           benchmark against other small TTS systems (results, scoring script)
docs/            prior_art.md
```

## Limitations

- English only, two voices (female and male); one voice per chip weights file.
- Speed is measured on one board only (ESP32-S3-DevKitC-1 N16R8 (rev v0.2, 8 MB octal PSRAM at 80 MHz, 16 MB flash, 240 MHz), 8 and 9 October 2026, firmware fedbd26; female voice on both days, male voice on 9 October); a second board or chip revision, cold power-on boots, the DAC/audio output stage and any listening, power draw and other supply/temperature conditions are not measured; the male voice and the repeat boots are the same single board, and the int4 and light sets were timed by hand only; the optimistic / central / pessimistic figures remain pre-board estimates.
- Stability on the board was observed, not proven: no corruption or crash in 600 golden-sentence iterations (about 2.5 hours) and a 21-minute synthetic two-core stress, on one board, one chip revision, USB powered, female voice only, no DAC; not covered are a second board, cold power-on boots, supply, power, temperature and runs of days. The Oído project reports faults with concurrent two-core PIE kernels on this chip; Ito did not reproduce them, but test dual-core on your own boards.
- The chip uses one fixed speaking style. Expressiveness comes from the text through the front, but there is no
  per-sentence style control.
- The pitch range is still slightly narrower than the teacher's (0.94 female, 0.97 male).
- Requires an ESP32-S3 with 8 MB PSRAM (N16R8 recommended); the weights alone are 3.8 MB.
- Grapheme-to-phoneme runs on the host (espeak-ng), not on the chip.
- Ito's output is synthetic speech in the voice of a real (LibriTTS-R) speaker. Please disclose that it is synthetic
  (see [`NOTICE`](NOTICE)).

## License

Ito has two parts, under two licenses:

- **Code** (`ito/`, `esp32/`, `eval/`): **GNU GPL v3** ([`LICENSE`](LICENSE)). The GPL allows commercial use *of the
  code*. For products that cannot meet GPLv3 terms (for example, devices that do not let users install modified
  firmware), Lokutor offers commercial licenses: see [`COMMERCIAL.md`](COMMERCIAL.md).
- **The voice models** (female voice: `ito_female.pt`, `ito_female_esp32s3.bin`; male voice: `ito_male.pt`, `ito_male_esp32s3.bin`; distributed only through
  [Hugging Face](https://huggingface.co/lokutor-ai/ito), and the Ito audio in `samples/` and
  `results/`): **CC BY-NC-SA 4.0** ([`models/LICENSE-WEIGHTS`](models/LICENSE-WEIGHTS),
  [legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)), plus Lokutor's
  [terms of use](models/TERMS.md). The GPL on the code does **not** cover the model. **Any commercial use of the
  model needs a written license from Lokutor.** That includes products, paid services, internal business use, and
  generating audio for commercial content. The same applies to weights that are fine-tuned, distilled, quantized or
  otherwise derived from Ito, and the terms of use exclude training commercial TTS models on Ito's output.
- Attributions for the teacher model and the data are in [`NOTICE`](NOTICE). Ito's output is synthetic speech in the
  voice of a real (LibriTTS-R) speaker: please say so when you share it.

**We'd love to hear from you.** Lokutor is open to collaborations. We offer no-cost licenses for small companies,
startups, makers selling small batches, education and research, and we also have other voices, other languages and
speech recognition for the same chip. Write to **contact@lokutor.com**.
