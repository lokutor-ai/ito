# Ito: natural-sounding streaming text-to-speech for a $5 chip

[![Ito demo: click to watch the video with sound](docs/demo.gif)](https://lokutor-ai.github.io/ito/)

*Click the animation to watch the 50-second video **with sound** (the GIF is silent). Every voice in it is Ito's own output from the chip engine.*

Ito is English text-to-speech that runs entirely on an **ESP32-S3** (240 MHz dual-core Xtensa LX7, 8 MB PSRAM, 16 MB
flash), with no cloud and no neural accelerator. It streams: audio starts after a short first chunk (125 ms) is ready, and the work
before it does not grow with the sentence length. The voice is distilled from a large open TTS model into 3.3 M parameters. Built by
[Lokutor](https://lokutor.com), the makers of [Oído](https://github.com/lokutor-ai/oido) (speech recognition on the
same chip).

> **Status (5 October 2026).** The on-chip engine is verified on a laptop and in Espressif's QEMU emulator: the
> firmware's audio is bit-identical to the host build of the engine, and every sample in [`samples/`](samples) is that
> engine's exact output. **Nothing has run on a physical board yet.** Time to first audio and real-time factor are
> **estimated** from exact instruction counts (QEMU) and an assumed PSRAM bandwidth. **Real-time playback is not established on silicon:**
> with the main weight set the optimistic and central estimates are faster than real time (RTF 0.43–0.47 and 0.63–0.66) and the pessimistic one is only just below it
> (0.97–0.99 for whole sentences, 0.96 for long ones). The vocoder is now 192 wide instead of 256, which is what moved the numbers; in a blind test
> (#10, one listener) it sounded the same as the 256-wide one. The engine starts with a 125 ms first chunk and a short ramp so that speech is gapless from the first chunk
> (estimated first sound 124–132 / 171–180 / 251–260 ms, optimistic / central / pessimistic). An earlier version of this page claimed 130–210 ms and
> real time at 0.5–1 GOPS; that was too optimistic (see [`esp32/README.md`](esp32/README.md) §4).
>
> **The firmware now enforces real time by measurement.** Each voice ships three weight sets (main int8; main with int4 blocks; a light set with one block fewer): at boot the board measures the real time of every chunk with each set, keeps the first
> whose measured RTF is <= 0.85, plans the playback start delay from those measurements, and prints `WARNING: DEGRADED` (and sets a flag) if even the fastest set measures >= 0.95. In the pessimistic estimate the light set would be kept as
> "marginal" (RTF 0.88, start delay about 0.41 s). This is tested in QEMU with injected timings, not on silicon: it is a guarantee **by measurement on the board that runs it**, not a promise about hardware we have not seen. We will publish board measurements here as soon as we have them.

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
| PSRAM / SRAM | peak 5.3 of 8 MB PSRAM, 320 of 379 KB internal SRAM (QEMU, final firmware) |
| Work before the first audio (125 ms first chunk) | **23.0–24.3 M instructions and 3.9 MB of weights read from PSRAM, for any sentence length** (exact counts from QEMU; 16.0–16.5 M and 3.45 MB for a 25 ms first chunk) |
| Work per second of audio | 76–86 M instructions on the dual-core critical path, 10.5–11.2 MB of weights read from PSRAM (24-frame chunks; exact counts; the conversion to time is an estimate) |
| Time to first audio | **estimated, not measured:** 124–132 ms optimistic, **171–180 ms central**, 251–260 ms pessimistic. The first chunk is 125 ms of audio and the chunks behind it are sized so that playback can start with it without a gap (the version of 2 October made 25 ms of sound after 145 ms and then fell silent for about 200 ms; `first 2` still does that; the first public version of the engine: 338 / 463 / 672 ms; the 256-wide vocoder of 3 October: 137–143 / 200–207 / 311–318 ms) |
| Real-time factor | **estimated, not measured:** 0.43–0.47 optimistic, **0.63–0.66 central** (0.65 for long sentences), 0.97–0.99 pessimistic (0.96 for long sentences). Below 1 is faster than real time; the pessimistic case (40 MB/s PSRAM, nothing overlapped, CPI 1.6) is **only just below it, which is not a margin**. With the 256-wide vocoder it was 0.77–0.80 central and 1.18–1.27 pessimistic |
| Start delay for gapless speech | **estimated:** optimistic = the time to first audio (125–130 ms), **central 176–182 ms**, pessimistic 558–663 ms (short sentences need the most); the firmware now plans this delay from its own measurements. With the 256-wide vocoder: 137–143 / 215–240 / 880–2200 ms. `delay <ms>` on the serial console holds playback for a chosen time |
| On a laptop | RTF ≈ 0.01 on an Apple M4 Max CPU, for both the PyTorch model (whole utterance) and the C engine |

All of this is **estimated from exact instruction counts, not measured on silicon.** The open question is the effective PSRAM
bandwidth and how much of it overlaps compute (GDMA); the pessimistic column is decided by those two assumptions. A weight pass
serves 24-frame (300 ms) chunks, and the 192-wide vocoder cuts the weight traffic from 15 to 11 MB per second of audio. The firmware's boot benchmark measures the real numbers and prints `BOARD_SUMMARY`, `CALIB` and `TIER_SELECT`.
The estimate model and full tables are in [`esp32/README.md`](esp32/README.md).

**The three weight sets, estimated the same way** (optimistic / central / pessimistic; whole sentences with the start-up ramp; not measured on silicon):

| | main int8 | main int4 | light (4 blocks, int4) |
|---|---|---|---|
| time to first audio | 124–132 / 171–180 / 251–260 ms | 127–133 / 168–174 / 236–243 ms | 118–124 / 157–164 / 222–229 ms |
| real-time factor | 0.43–0.47 / 0.63–0.66 / 0.97–0.99 | 0.45–0.48 / 0.62–0.64 / 0.92–0.93 | 0.42–0.45 / 0.59–0.61 / 0.88 |
| gapless start delay | 125–130 / 176–182 / 558–663 ms | 127–133 / 173–179 / 481–526 ms | 118–124 / 162–169 / 406–429 ms |

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
- Speed is estimated until board measurements are published; the boot calibration that picks a weight set and plans the start delay has only been tested with simulated timings.
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
