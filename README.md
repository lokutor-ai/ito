# Ito: natural-sounding streaming text-to-speech for a $5 chip

Ito is English text-to-speech that runs entirely on an **ESP32-S3** (240 MHz dual-core Xtensa LX7, 8 MB PSRAM, 16 MB
flash), with no cloud and no neural accelerator. It streams: audio starts after a short first chunk (25 ms) is ready, and the work
before it does not grow with the sentence length. The voice is distilled from a large open TTS model into 4.4 M parameters. Built by
[Lokutor](https://lokutor.com), the makers of [Oído](https://github.com/lokutor-ai/oido) (speech recognition on the
same chip).

> **Status (2 October 2026).** The on-chip engine is verified on a laptop and in Espressif's QEMU emulator: the
> firmware's audio is bit-identical to the host build of the engine, and every sample in [`samples/`](samples) is that
> engine's exact output. **Nothing has run on a physical board yet.** Time to first audio and real-time factor are
> **estimated** from exact instruction counts (QEMU) and an assumed PSRAM bandwidth. **Real-time playback is not established:**
> in the central estimate the board is slightly slower than real time. An earlier version of this page claimed 130–210 ms and
> real time at 0.5–1 GOPS; that was too optimistic (see [`esp32/README.md`](esp32/README.md) §4). The firmware measures the real numbers at boot;
> we will publish them here after we run it on boards.

> **Licensing in one line.** The code is GPLv3; the voice model that makes it talk is **non-commercial**
> (CC BY-NC-SA 4.0 + [terms](models/TERMS.md)). Hobby, research and education use is welcome. For anything commercial,
> or if you are a small team that wants to build with it, [talk to us](#license): we like collaborating.

**Hear it:** [samples next to sanoTTS and the teacher, with a blind mode](https://lokutor-ai.github.io/ito/) ·
**Type your own:** [Hugging Face Space](https://huggingface.co/spaces/lokutor-ai/ito-tts) (runs the chip engine,
bit-exact, on a CPU) · **Weights:** [lokutor-ai/ito-tts-v3](https://huggingface.co/lokutor-ai/ito-tts-v3)

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

The float-model clips used in the listening test (sentences 01, 03, 04 and 06 were rated), next to the teacher's, are in
[`results/blind9/audio/`](results/blind9/audio).

## Listening tests

**Blind test #9** (2 October 2026). There were 24 clips in random order with the system names hidden, rated 1–5 for
naturalness. Each system read the same four new conversational sentences (a question, a long sentence, hard words, and
a question with hard words).

| System | Parameters | Runs on | Mean score (4 clips) |
|---|---|---|---|
| Teacher: StyleTTS 2 (LibriTTS model, reference voice) | large (diffusion, PL-BERT) | GPU / laptop | 4.75 |
| **Ito**, streaming front (this release) | **4.4 M** | **ESP32-S3** | **4.00** (4/4/4/4) |
| Ito variant with a bidirectional front (no streaming text side) | 4.5 M | ESP32-S3 | 4.00 |
| Ito variant with a wider vocoder | 8.2 M | over the chip budget | 3.75 |
| sanoTTS "amy" | 1.46 M | — | 2.00 |
| sanoTTS "heart-nano" | 0.29 M | — | 1.00 |

Please read these numbers with their limits:
- **There was one listener**, an expert (Lokutor's founder), and **four sentences per system**. This is a strong
  direction, not a statistically powered MOS study. With n = 4, differences under about half a point are noise.
- **The rated Ito clips are the float PyTorch model**, through the same streaming path as the chip. They used a small
  optional text-to-style predictor. The shipped chip uses a fixed mean style instead, so that the time to first audio
  does not grow with sentence length. In a paired comparison on held-out sentences, the predictor changed the log-mel
  distance to the reference by only −0.009. The quantised engine scores PESQ 4.52 against the float model. **The exact
  chip configuration has not been rated in a blind test yet.**
- All clips were trimmed and loudness-normalised to −20 LUFS at 24 kHz. Ratings, key, sentences and automatic metrics
  are in [`results/blind9/`](results/blind9).

Earlier, **blind test #8** (one listener, four held-out sentences) compared two earlier Ito variants with the teacher.
The scores were teacher 4.00, variant A 3.25, variant B 3.00. In #6
and #7 the gap to the teacher had been 2.25 to 2.5 points, and in #9 it is 0.75. Details are in
[`results/blind8/`](results/blind8).

**Automatic metrics** on the eight sentences above (UTMOS; Whisper word error rate after number normalisation):

| | Teacher | Ito | sanoTTS amy | sanoTTS heart-nano |
|---|---|---|---|---|
| UTMOS | 4.49 | 4.46 | 3.98 | 2.07 |
| WER, Whisper medium.en / base | 0 / 0 | 0 / 0 | 0 / 1.0 % | 1.0 / 1.0 % |

UTMOS cannot hear intonation, so treat it as a check, not a verdict. On 60 held-out sentences, Ito's pitch range is
0.94 of the teacher's: the per-utterance spread of log-F0 relative to the teacher.

**Benchmark against other small and embedded TTS systems:** see [`bench/`](bench) *(being merged)*.
**Prior art:** see [`docs/prior_art.md`](docs/prior_art.md) *(being merged)*.

## Size and compute

| | |
|---|---|
| Parameters | 4.40 M: acoustic front 1.62 M + vocoder 2.79 M |
| Chip weights | **4.89 MB** (`ito_v3_esp32s3.bin`): int8 mel head and vocoder; int16 pitch path |
| Flash | 296 KB app + 4.89 MB weights; about 9 MB of the 14 MB weights partition stays free |
| PSRAM / SRAM | peak 5.9 of 8 MB PSRAM, 327 of 383 KB internal SRAM (measured in QEMU) |
| Work before the first audio (25 ms first chunk) | **23 M instructions and 4.5 MB of weights read from PSRAM, for any sentence length** (exact counts from QEMU) |
| Work per second of audio | 133–144 M instructions on the dual-core critical path, 38 MB of weights read from PSRAM (exact counts; the conversion to time is an estimate) |
| Time to first audio | **estimated, not measured:** 124–127 ms optimistic, **176–179 ms central**, 266–269 ms pessimistic (the first public version of the engine: 338 / 463 / 672 ms) |
| Real-time factor | **estimated, not measured:** 0.72–0.78 optimistic, **1.15–1.23 central**, 1.9–2.0 pessimistic. Below 1 is real time, so **real time is not established**; the CPU alone is 0.8–0.9, and reading the weights from PSRAM is the bottleneck |
| On a laptop | RTF ≈ 0.01 on an Apple M4 Max CPU, for both the PyTorch model (whole utterance) and the C engine |

The open question is the effective PSRAM bandwidth and how much of it overlaps compute (GDMA). The identified fixes if it is too slow
are working GDMA overlap, a leaner scratch so 16-frame chunks fit (half the weight traffic), and fewer float operations. The firmware's boot benchmark measures the real numbers
and prints `BOARD_SUMMARY`. The estimate model and full tables are in [`esp32/README.md`](esp32/README.md).

## How it works

```
 text ─► espeak-ng phonemes (on the host) ─► token ids ─────────────────────────────────────── over serial to the chip
                                                 │
   ┌──────────────────────── acoustic front (1.62 M) ────────────────────────┐   ┌────────── vocoder (2.79 M) ─────────┐
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
  lookahead over features derived from the text, which is already known, so it adds no audio latency. Each 100 ms
  chunk is computed from ring buffers. The streamed output is bit-identical to whole-utterance output on the chip, and equal to float precision (about 120 dB) in PyTorch.
- **The engine** (`esp32/engine`) is new portable C99 code. It uses int8 and int16 GEMMs on the S3's vector unit
  through esp-nn, per-row activation scales (so streaming stays bit-exact), its own transcendental functions
  built from basic arithmetic (bit-identical on host and chip), and a mixed-radix FFT. It reads weights from PSRAM with tiled
  staging and can use both cores.

Training code and recipe are not public; contact us for research collaborations (contact@lokutor.com).

## Quickstart

**Get access to the voice model (once).** The weights are distributed through the gated Hugging Face repository
[lokutor-ai/ito-tts-v3](https://huggingface.co/lokutor-ai/ito-tts-v3). Open it, accept the
[terms](models/TERMS.md) (free for research, education and personal projects), then log in on your machine:

```bash
pip install -e .                 # also installs huggingface_hub
huggingface-cli login            # or: hf auth login, or export HF_TOKEN=...
```

Everything below downloads the weights it needs on first use. Details and offline use: [`models/README.md`](models/README.md).

**Python (CPU is fine).** Needs Python ≥ 3.9.

```bash
ito-tts "Good morning! The coffee is ready." -o hello.wav              # voice D (female, default)
ito-tts --voice g "Good morning! The coffee is ready." -o hello_g.wav  # voice G (male)
```

Two voices: **D** (female, LibriTTS-R speaker 4970; `ito_v3.pt` / `ito_v3_esp32s3.bin`) and **G** (male, LibriTTS-R
speaker 5105; `ito_v3_G.pt` / `ito_v3_G_esp32s3.bin`). Each chip file holds one voice.

```python
from ito import Ito
tts = Ito.load()                                   # voice D; downloads ito_v3.pt on first use
tts_g = Ito.load(voice="g")                        # voice G (male): ito_v3_G.pt
wav = tts.synthesize("Could you grab some quinoa on your way home?")   # float32 numpy, 24 kHz
for chunk in tts.stream("A sentence of any length."):                  # 100 ms chunks, as on the chip
    ...
```

`--style predicted` uses the optional text-to-style predictor, as in the blind test (voice D only). `--stream` uses the
chunked path.

**On a laptop, with the chip's exact arithmetic** (no board):

```bash
cd esp32/host && make && make test && cd ../..      # make test fetches ito_v3_esp32s3.bin into models/
python esp32/tools/chip_wav.py "Good morning! The coffee is ready." hello_chip.wav
python esp32/tools/chip_wav.py --voice g "Good morning! The coffee is ready." hello_chip_g.wav   # voice G
```

`make test VOICE=g` runs the same host test on voice G's chip file.

**On a board.** You need an ESP32-S3-DevKitC-1 **N16R8** and an I2S DAC or amplifier: BCLK→GPIO15, LRCK→GPIO16,
DIN→GPIO17 (PCM5102A or MAX98357A).

```bash
pip install esptool pyserial
esp32/tools/fetch_weights.sh                       # ito_v3_esp32s3.bin from Hugging Face into models/ (flash.sh also does it)
esp32/tools/flash.sh /dev/ttyUSB0                  # prebuilt app + voice D (ito_v3_esp32s3.bin)
VOICE=g esp32/tools/flash.sh /dev/ttyUSB0          # ... or voice G (male): flashes ito_v3_G_esp32s3.bin at 0x200000
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # press RST: self-test, benchmark, 3 demo sentences, READY
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
results/         blind9/ and blind8/ (listening tests), chip/ (host tests, QEMU log, op profile, estimates)
bench/           benchmark against other small TTS systems (being merged)
docs/            prior_art.md (being merged)
```

## Limitations

- English only, two voices (D female, G male); one voice per chip weights file.
- Speed is estimated until board measurements are published.
- The chip uses one fixed speaking style. Expressiveness comes from the text through the front, but there is no
  per-sentence style control.
- The pitch range is still slightly narrower than the teacher's (0.94).
- Requires an ESP32-S3 with 8 MB PSRAM (N16R8 recommended); the weights alone are 4.9 MB.
- Grapheme-to-phoneme runs on the host (espeak-ng), not on the chip.
- Ito's output is synthetic speech in the voice of a real (LibriTTS-R) speaker. Please disclose that it is synthetic
  (see [`NOTICE`](NOTICE)).

## License

Ito has two parts, under two licenses:

- **Code** (`ito/`, `esp32/`, `eval/`): **GNU GPL v3** ([`LICENSE`](LICENSE)). The GPL allows commercial use *of the
  code*. For products that cannot meet GPLv3 terms (for example, devices that do not let users install modified
  firmware), Lokutor offers commercial licenses: see [`COMMERCIAL.md`](COMMERCIAL.md).
- **The voice models** (voice D: `ito_v3.pt`, `ito_v3_esp32s3.bin`; voice G: `ito_v3_G.pt`, `ito_v3_G_esp32s3.bin`; distributed only through
  [Hugging Face](https://huggingface.co/lokutor-ai/ito-tts-v3), and the Ito audio in `samples/` and
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
