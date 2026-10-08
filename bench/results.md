# Ito bench v1.2: small and edge English TTS (2026-10-08)

**What is compared.** Ito as it ships on the ESP32-S3 (the 192/576, 5-block vocoder; the firmware's own C engine, host build:
its PCM is bit-identical to the chip, verified in QEMU), its teacher, and 21 public small / on-device systems (25 systems
in `results.json` in all), on 54 fixed English prompts (`prompts.json`: conversational lines, questions, numbers/dates, hard
words, long sentences; none occurs in Ito's training corpus, checked by phoneme-string match). All audio goes through
identical post-processing before scoring. **Headline row: Ito 192 vocoder, chip-exact, female voice** (main int8 weight set,
the released `ito_female_esp32s3.bin`); all paired deltas are against it. Also scored: the male voice (no teacher row for it:
the teacher is StyleTTS 2 in the female speaker's voice), the int4 and light weight sets of both voices, and, kept for
transparency, the rows of the first release (256-wide vocoder; 4.05 M parameters on the chip, 4.89 MB blob).

**v1.2 changes (2026-10-08):** the Ito rows were re-rendered and re-scored with the shipped model (`ito_192_*` rows). The
first-release rows (`ito_v3_chip_fefix`, `ito_v3_chip`, `ito_v3_float`) are unchanged and no longer the headline. The
speed/size fields of the old rows (4.89 MB, 83.6 M ops, ~350 M MACs/s, "TTFA 212 ms @0.5 GOPS") describe the 256-wide model and
are superseded; the shipped model's estimates are in `esp32/README.md`. The Inflect Nano size is 3.96 M (Owen Song's figure,
3,966,721 parameters), and the public systems are 21, not 22 (the 22 counted the teacher). All other systems' scores are
the stored ones (not re-run); their audio is not re-rendered.

**v1.1 changes:** added Inflect Nano/Micro v2, TinyTTS (pschatzmann port), Pocket TTS, Supertonic 2/3, MOSS-TTS-Nano;
Ito re-rendered with the fixed text frontend (`normalize()`: decimals / dollars / thousands; only prompts 31 and 37 change).
Not added, with reason: **NeuTTS Air/Nano** (every checkpoint, including the Air GGUFs, is gated on Hugging Face:
licence acceptance + login, which the owner must do); **Piper x_low** (no English x_low voice exists; the 14 x_low voices
are ca/de/es/it/kk/ne/nl/uk/vi/zh); **Grovety tinyTTS / HxTTS** (closed model, needs its Himax NPU kit); **Moonshine
Micro neural-tts** (host CLI is public but the voice pack is not); **SVOX Pico** (our 64-bit build produced empty audio
nondeterministically). tronghieuit/tiny-tts is the same model as the pschatzmann port (one row).

## Findings (read with the CIs)

- **Shipped model, female voice:** UTMOSv2 3.21 [3.12, 3.29], UTMOS22 4.43, DNSMOS 3.34, WER 0.6 % (CER 0.2 %). The
  first-release (256-wide) row measured 3.15 / 4.44 / 3.39 / 0.4 %. Paired, 192 minus 256: UTMOSv2 +0.06 [−0.02, 0.14]
  (not significant), UTMOS22 −0.01 [−0.03, 0.00], **DNSMOS −0.05 [−0.07, −0.03]** (small but outside the interval), WER +0.1 pts
  [0.0, 0.5] (one more error: "9th"→"9" in prompt 2). So the narrower vocoder is indistinguishable on the three
  naturalness predictors and WER, and slightly lower on DNSMOS. UTMOSv2 is stochastic (see Method): treat differences under about 0.05 as noise.
- **Male voice** (no teacher row): UTMOSv2 3.26 [3.16, 3.34], UTMOS22 4.41, DNSMOS 3.43, WER 0.7 %. Different speaker, so
  differences to the female row (UTMOSv2 +0.04 [−0.05, 0.14] n.s., UTMOS22 −0.02, DNSMOS +0.08, F0 SD 2.42 vs 2.80 st) are not
  a model comparison. The other systems are single-speaker too; every cross-system statement below is for the female voice
  unless it says male.
- **Weight sets** (paired vs the main int8 set of the same voice): female int4 / light: UTMOSv2 −0.08 [−0.16, −0.00] /
  −0.09 [−0.17, −0.02] (borderline), UTMOS22 −0.01 / −0.01 (n.s.), DNSMOS 0.00 / +0.03, WER identical. Male int4 / light:
  UTMOSv2 0.00 [−0.07, 0.07] / −0.02 [−0.10, 0.06], UTMOS22 −0.00 / −0.02 [−0.04, −0.00], DNSMOS −0.00 / −0.04 [−0.06, −0.02], WER
  identical. So the fallback sets cost at most a few hundredths on the predictors (the audio is deterministic; they are post-training
  quantizations of the same model). The first-release int8-vs-float comparison remains: no measurable difference.
- **MCU-class systems (measured or emulated on a microcontroller):** vs **Inflect Nano v2** (3.96 M, ESP32-P4, ~3.5x slower
  than real time) Ito female is **slightly ahead on UTMOSv2** (+0.13 [0.03, 0.23]; the lower bound is close to 0 and UTMOSv2
  is noisy), **tied on UTMOS22** (4.43 vs 4.41; −0.02 [−0.05, 0.01]), **behind on DNSMOS** (3.34 vs 3.40; Inflect +0.06
  [0.03, 0.08]) and more intelligible (WER 0.6 vs 1.1 %, Δ 0.6 pts [0.1, 1.1]). With the male voice Ito is ahead on UTMOSv2 (+0.18 [0.08, 0.27]) and DNSMOS
  (+0.03 [0.00, 0.06]), tied on UTMOS22 (Δ 0.00 [−0.03, 0.03]), WER 0.7 vs 1.1 % (Δ 0.4 [0.0, 0.9]). vs **TinyTTS** (1.6 M, ESP32-S3,
  22.9x slower than real time): +0.76 UTMOSv2, WER 0.6 vs 6.8 %. vs **sanoTTS heart-nano** (0.29 M): +1.88. Ito is the only one
  of these that streams; its real-time figure is still an estimate (no silicon yet).
- **Models that beat Ito female on UTMOSv2** (all ≥ 9 M params, none MCU-class; CIs exclude 0): Kokoro-82M (+0.66),
  Supertonic 3 (+0.63), Piper lessac medium (+0.48), Supertonic 2 (+0.41), **Inflect Micro v2 (+0.25, 9.36 M)**,
  MOSS-TTS-Nano (+0.22), our teacher (+0.22, the model Ito was distilled from), Piper amy low (+0.21) and Piper lessac low (+0.18); Pocket TTS (+0.10 [−0.01, 0.21]) is
  tied. On UTMOS22 Ito (4.43) ties Inflect Nano/Micro, Supertonic 2/3, Piper amy/lessac low and is above Piper lessac medium,
  Pocket and MOSS; only Kokoro (+0.06) and the teacher (+0.04) are higher. On DNSMOS OVRL Ito (3.34) **is below Inflect Nano,
  Inflect Micro and Kokoro (by 0.04–0.06)**, ties the teacher, Piper lessac low/medium, Supertonic 2/3 and Pocket, and is above Piper amy low, MOSS, MeloTTS, TinyTTS, sanoTTS, Kitten (nano
  borderline), eSpeak and Flite.
- **WER:** Ito (0.6 %) and eSpeak NG (0.3 %; difference not significant) are the most intelligible systems; Ito is
  significantly better than 10 of the 22 others (paired CIs exclude 0): Inflect Nano, Supertonic 2/3, Pocket TTS, MOSS, Piper lessac low,
  MeloTTS, Flite, TinyTTS and sanoTTS kristin. vs Inflect Micro, Kokoro, the teacher, Piper lessac medium / amy low, sanoTTS amy / heart / heart-nano and Kitten it is not
  significant (Inflect Micro and sanoTTS heart / heart-nano have a lower bound of exactly 0.0 and are counted as not significant). Worst: MeloTTS 3.2 %, TinyTTS and sanoTTS kristin 6.8 %.
- **Prosody proxy:** Ito female's F0 SD (2.80 st) is close to its teacher (2.84) and Kokoro (2.87); Inflect, Pocket, Piper lessac
  and Kitten are wider (3.2–4.4 st), sanoTTS and MOSS flatter (2.2–2.6). The male voice's 2.42 st is a different
  speaker's. A flatness check, not a naturalness score.
- **Predictors disagree** for some systems (Kitten: UTMOS22 ≈ 4.0 vs UTMOSv2 ≈ 1.9). The earlier blind test #9 (one
  listener, 4 sentences) had teacher 4.75, Ito 4.00, sanoTTS amy 2.00. Run `listening_test/` before claiming
  naturalness against Kokoro/Supertonic/Piper/Inflect Micro.
- **Speed figures are not like-for-like:** Ito's TTFA/RTF are estimates from exact instruction counts and an assumed PSRAM
  bandwidth (shipped model: 69.7 M int8 MACs before the 8-frame first chunk of the host tool, 124–132 / 171–180 / 251–260 ms
  to first audio and RTF 0.43–0.47 / 0.63–0.66 / 0.97–0.99 optimistic / central / pessimistic, `esp32/README.md`; ~263 M int8
  MACs per second of audio). Inflect Nano (P4) and TinyTTS (S3/P4) MCU numbers are third-party measurements from their
  READMEs; sanoTTS publishes MCU timing only for its 567 K voice.

## Results

54 prompts (`prompts.json`), every system scored on identical post-processed audio. Mean [95% bootstrap CI over prompts]. WER/CER in %, corpus-level, Whisper large-v3, normalised text. F0 SD = voiced pitch standard deviation in semitones (prosody-variability proxy; more is not automatically better). The `Ito 256 vocoder` rows are the first release, kept for transparency.

| system | params | MCU | UTMOSv2 | UTMOS22 | DNSMOS OVRL | WER % | CER % | F0 SD (st) | TTFA / RTF on MCU |
|---|---|---|---|---|---|---|---|---|---|
| **Ito 256 vocoder (first release) chip-exact (frontend fix)** (ours) | 4.05 M (int8/int16) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.15 [3.06, 3.23] | 4.44 [4.43, 4.46] | 3.39 [3.36, 3.42] | 0.4 [0.0, 0.9] | 0.1 [0.0, 0.3] | 2.89 [2.77, 3.02] | size/speed fields describe the 256-wide first release (4.89 MB blob, 83.6 M MACs before first chunk, ~350 M MACs/s); superseded by the 192 rows |
| **Ito 256 vocoder (first release) chip-exact** (ours, bench v1 frontend) | 4.05 M (int8/int16) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.18 [3.09, 3.27] | 4.44 [4.43, 4.45] | 3.39 [3.35, 3.42] | 1.4 [0.1, 3.8] | 1.2 [0.0, 3.8] | 2.88 [2.77, 3.01] | size/speed fields describe the 256-wide first release (4.89 MB blob, 83.6 M MACs before first chunk, ~350 M MACs/s); superseded by the 192 rows |
| **Ito 192 vocoder, chip-exact, female** (ours, shipped) | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.21 [3.12, 3.29] | 4.43 [4.41, 4.45] | 3.34 [3.31, 3.37] | 0.6 [0.1, 1.1] | 0.2 [0.0, 0.4] | 2.80 [2.71, 2.89] | est. TTFA 124-132 / 171-180 / 251-260 ms, RTF 0.43-0.47 / 0.63-0.66 / 0.97-0.99 (optimistic / central / pessimistic; instruction counts + assumed PSRAM bandwidth, not silicon) |
| **Ito 192 vocoder, chip-exact, male** (ours, shipped; no teacher row for this voice) | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.26 [3.16, 3.34] | 4.41 [4.39, 4.42] | 3.43 [3.41, 3.45] | 0.7 [0.1, 1.3] | 0.2 [0.1, 0.5] | 2.42 [2.33, 2.52] | est. TTFA 124-132 / 171-180 / 251-260 ms, RTF 0.43-0.47 / 0.63-0.66 / 0.97-0.99 (optimistic / central / pessimistic; instruction counts + assumed PSRAM bandwidth, not silicon) |
| Ito 192, int4 set, chip-exact, female | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.13 [3.04, 3.21] | 4.42 [4.39, 4.44] | 3.34 [3.31, 3.37] | 0.6 [0.1, 1.1] | 0.2 [0.0, 0.4] | 2.81 [2.73, 2.90] | est. TTFA 124-132 / 171-180 / 251-260 ms, RTF 0.43-0.47 / 0.63-0.66 / 0.97-0.99 (optimistic / central / pessimistic; instruction counts + assumed PSRAM bandwidth, not silicon) |
| Ito 192, light set, chip-exact, female | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.12 [3.03, 3.20] | 4.42 [4.40, 4.43] | 3.37 [3.34, 3.40] | 0.6 [0.1, 1.1] | 0.2 [0.0, 0.4] | 2.82 [2.72, 2.94] | estimated, see esp32/README.md (not silicon) |
| Ito 192, int4 set, chip-exact, male | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.26 [3.17, 3.34] | 4.40 [4.39, 4.42] | 3.42 [3.40, 3.45] | 0.6 [0.1, 1.1] | 0.2 [0.0, 0.4] | 2.40 [2.33, 2.47] | est. TTFA 124-132 / 171-180 / 251-260 ms, RTF 0.43-0.47 / 0.63-0.66 / 0.97-0.99 (optimistic / central / pessimistic; instruction counts + assumed PSRAM bandwidth, not silicon) |
| Ito 192, light set, chip-exact, male | 3.34 M (2.99 M stored on chip as int8/int16/f32) | yes, ESP32-S3 (QEMU bit-exact; silicon pending) | 3.23 [3.15, 3.31] | 4.39 [4.36, 4.41] | 3.39 [3.37, 3.41] | 0.6 [0.1, 1.1] | 0.2 [0.0, 0.4] | 2.44 [2.33, 2.59] | estimated, see esp32/README.md (not silicon) |
| Ito 256 vocoder (first release) float (ours) | 4.40 M (fp32) | no (float reference of the chip model) | 3.20 [3.10, 3.29] | 4.44 [4.43, 4.45] | 3.40 [3.37, 3.42] | 1.4 [0.1, 3.8] | 1.2 [0.0, 3.8] | 2.93 [2.79, 3.08] | – |
| StyleTTS 2 LibriTTS, voice D (our teacher) | 191 M (all loaded modules) | no | 3.43 [3.34, 3.52] | 4.47 [4.44, 4.48] | 3.34 [3.28, 3.38] | 1.5 [0.1, 4.0] | 1.3 [0.1, 3.9] | 2.84 [2.73, 2.96] | – |
| sanoTTS amy | 1.45 M | no (paper: not run on MCU) | 2.80 [2.72, 2.88] | 3.96 [3.89, 4.04] | 3.18 [3.12, 3.23] | 1.2 [0.3, 2.6] | 0.5 [0.1, 0.9] | 2.42 [2.13, 2.73] | n/p |
| sanoTTS kristin | 1.40 M | no (paper: not run on MCU) | 1.98 [1.89, 2.06] | 3.56 [3.49, 3.63] | 3.11 [3.05, 3.16] | 6.8 [4.1, 9.8] | 3.4 [1.9, 5.1] | 2.15 [1.96, 2.35] | n/p |
| sanoTTS heart | 2.27 M | no published MCU run | 1.95 [1.88, 2.01] | 3.19 [3.12, 3.26] | 3.14 [3.08, 3.20] | 1.7 [0.5, 3.3] | 0.6 [0.2, 1.2] | 2.57 [2.51, 2.64] | n/p |
| sanoTTS heart-nano | 0.29 M (int8) | MCU-sized; no published timing for this voice | 1.33 [1.27, 1.39] | 2.17 [2.08, 2.27] | 2.97 [2.90, 3.04] | 1.7 [0.5, 3.1] | 0.7 [0.2, 1.4] | 2.26 [2.18, 2.35] | published for sanoTTS's 567 K 'robot' voice only: RTF 0.383 measured on ESP32-S3 (repo README; paper: 0.22x RT, ~45 MMAC/s) |
| Inflect Nano v2 (Owen Song) | 3.96 M | yes, ESP32-P4 (third-party port, whole-utterance) | 3.08 [3.01, 3.15] | 4.41 [4.38, 4.44] | 3.40 [3.38, 3.42] | 1.1 [0.4, 1.8] | 0.4 [0.2, 0.7] | 3.79 [3.59, 3.97] | ESP32-P4 @360 MHz: RTF ~3.5 (0.47 s audio in 1.66 s), inlanger/esp32-p4-inflect-tts README |
| Inflect Micro v2 (Owen Song) | 9.36 M | no published MCU run | 3.46 [3.37, 3.53] | 4.41 [4.37, 4.45] | 3.38 [3.35, 3.41] | 1.0 [0.4, 1.7] | 0.5 [0.2, 0.9] | 3.65 [3.46, 3.85] | – |
| TinyTTS (pschatzmann port) | ~1.6 M | yes, ESP32-S3 / P4 (whole-utterance) | 2.45 [2.39, 2.52] | 3.66 [3.59, 3.73] | 3.29 [3.26, 3.33] | 6.8 [4.2, 9.9] | 4.2 [2.5, 6.3] | 3.42 [3.30, 3.54] | ESP32-S3: 34.1 s for 1.49 s audio (22.9x slower than real time); P4: 28.4 s (19x), repo README |
| Piper lessac low | 15.7 M | no (ONNX on Linux; Raspberry Pi class) | 3.39 [3.33, 3.46] | 4.41 [4.38, 4.43] | 3.35 [3.31, 3.39] | 1.2 [0.5, 2.2] | 0.6 [0.2, 1.1] | 4.36 [4.18, 4.53] | – |
| Piper lessac medium | 15.7 M | no (ONNX on Linux) | 3.69 [3.63, 3.76] | 4.28 [4.22, 4.34] | 3.29 [3.23, 3.34] | 0.7 [0.1, 1.4] | 0.4 [0.1, 0.9] | 4.42 [4.27, 4.58] | – |
| Piper amy low | 15.6 M | no (ONNX on Linux) | 3.42 [3.34, 3.49] | 4.44 [4.41, 4.46] | 3.28 [3.23, 3.32] | 0.7 [0.1, 1.4] | 0.3 [0.1, 0.8] | 1.76 [1.67, 1.85] | – |
| Kitten TTS nano 0.8 | 14.0 M | no (ONNX CPU) | 1.99 [1.91, 2.07] | 3.93 [3.85, 4.01] | 3.31 [3.29, 3.33] | 1.1 [0.3, 2.2] | 0.2 [0.0, 0.4] | 4.16 [3.86, 4.46] | – |
| Kitten TTS micro 0.8 | 35.0 M | no (ONNX CPU) | 1.78 [1.71, 1.85] | 3.97 [3.91, 4.04] | 3.25 [3.22, 3.28] | 0.8 [0.1, 1.8] | 0.1 [0.0, 0.3] | 3.64 [3.38, 3.89] | – |
| Kitten TTS mini 0.8 | 73.2 M | no (ONNX CPU) | 1.88 [1.84, 1.93] | 4.02 [3.95, 4.09] | 3.27 [3.24, 3.30] | 1.1 [0.2, 2.5] | 0.3 [0.1, 0.8] | 3.21 [3.02, 3.41] | – |
| Kokoro-82M (af_heart) | 81.8 M | no | 3.87 [3.81, 3.93] | 4.49 [4.48, 4.50] | 3.41 [3.38, 3.43] | 0.8 [0.1, 1.8] | 0.2 [0.0, 0.4] | 2.87 [2.77, 3.00] | – |
| MeloTTS EN-US | 51.9 M | no | 3.03 [2.96, 3.09] | 3.72 [3.64, 3.80] | 3.01 [2.95, 3.07] | 3.2 [1.6, 5.1] | 1.4 [0.6, 2.5] | 2.85 [2.70, 3.01] | – |
| Pocket TTS (Kyutai, alba) | 110 M | no (CPU; ~6x RT on Apple M4) | 3.31 [3.23, 3.40] | 4.33 [4.29, 4.37] | 3.31 [3.27, 3.35] | 2.5 [1.2, 4.1] | 0.4 [0.1, 0.8] | 3.64 [3.40, 3.88] | – |
| Supertonic 2 (F1) | 65.5 M | no (CPU / mobile) | 3.62 [3.52, 3.71] | 4.44 [4.41, 4.47] | 3.35 [3.29, 3.40] | 2.4 [1.0, 4.1] | 1.3 [0.5, 2.4] | 2.81 [2.55, 3.07] | – |
| Supertonic 3 (F1) | 99.2 M | no (CPU / mobile) | 3.84 [3.78, 3.91] | 4.45 [4.43, 4.47] | 3.32 [3.28, 3.36] | 1.7 [0.6, 3.0] | 0.6 [0.2, 1.1] | 2.70 [2.57, 2.82] | – |
| MOSS-TTS-Nano (ONNX, en_2 ref) | ~100 M (vendor) | no (CPU) | 3.44 [3.37, 3.50] | 4.37 [4.33, 4.40] | 3.21 [3.15, 3.27] | 1.7 [0.4, 3.3] | 1.2 [0.2, 2.5] | 2.27 [2.13, 2.41] | – |
| eSpeak NG | rule-based | yes (community ESP32 ports, not verified here) | 1.74 [1.69, 1.80] | 2.14 [2.07, 2.21] | 2.76 [2.71, 2.80] | 0.3 [0.0, 0.7] | 0.1 [0.0, 0.3] | 1.61 [1.54, 1.67] | – |
| Flite (cmu_us_slt) | clustergen | yes (community ports, not verified here) | 1.55 [1.51, 1.60] | 2.59 [2.54, 2.64] | 2.59 [2.53, 2.65] | 1.8 [0.8, 3.0] | 1.0 [0.3, 1.7] | 1.09 [1.06, 1.12] | – |

Paired difference vs **Ito 192 vocoder, chip-exact, female** (ours, shipped) (system minus Ito chip-exact with the fixed frontend, same prompts; a CI that excludes 0 is a real difference):

| system | Δ UTMOSv2 | Δ UTMOS22 | Δ DNSMOS OVRL | Δ WER (pts) | Δ F0 SD |
|---|---|---|---|---|---|
| **Ito 256 vocoder (first release) chip-exact (frontend fix)** (ours) | -0.06 [-0.14, 0.02] | 0.01 [0.00, 0.03] | 0.05 [0.03, 0.07] | -0.1 [-0.5, 0.0] | 0.09 [-0.00, 0.19] |
| **Ito 256 vocoder (first release) chip-exact** (ours, bench v1 frontend) | -0.03 [-0.12, 0.07] | 0.01 [0.00, 0.03] | 0.04 [0.01, 0.07] | 0.8 [-0.4, 3.1] | 0.08 [-0.01, 0.19] |
| **Ito 192 vocoder, chip-exact, male** (ours, shipped; no teacher row for this voice) | 0.04 [-0.05, 0.14] | -0.02 [-0.04, -0.00] | 0.08 [0.05, 0.12] | 0.1 [0.0, 0.4] | -0.39 [-0.47, -0.30] |
| Ito 192, int4 set, chip-exact, female | -0.08 [-0.16, -0.00] | -0.01 [-0.03, 0.01] | -0.00 [-0.02, 0.01] | 0.0 [0.0, 0.0] | 0.01 [-0.00, 0.03] |
| Ito 192, light set, chip-exact, female | -0.09 [-0.17, -0.02] | -0.01 [-0.03, 0.01] | 0.03 [0.00, 0.05] | 0.0 [0.0, 0.0] | 0.02 [-0.05, 0.10] |
| Ito 192, int4 set, chip-exact, male | 0.04 [-0.04, 0.13] | -0.03 [-0.04, -0.01] | 0.08 [0.05, 0.11] | 0.0 [0.0, 0.0] | -0.40 [-0.47, -0.34] |
| Ito 192, light set, chip-exact, male | 0.02 [-0.07, 0.12] | -0.04 [-0.07, -0.02] | 0.04 [0.01, 0.08] | 0.0 [0.0, 0.0] | -0.36 [-0.46, -0.24] |
| Ito 256 vocoder (first release) float (ours) | -0.01 [-0.10, 0.08] | 0.01 [-0.01, 0.03] | 0.05 [0.02, 0.08] | 0.8 [-0.4, 3.1] | 0.12 [0.01, 0.27] |
| StyleTTS 2 LibriTTS, voice D (our teacher) | 0.22 [0.13, 0.31] | 0.04 [0.02, 0.06] | -0.01 [-0.06, 0.04] | 1.0 [-0.3, 3.3] | 0.04 [-0.09, 0.17] |
| sanoTTS amy | -0.41 [-0.51, -0.30] | -0.47 [-0.54, -0.39] | -0.16 [-0.21, -0.12] | 0.7 [-0.3, 2.0] | -0.38 [-0.70, -0.05] |
| sanoTTS kristin | -1.23 [-1.35, -1.12] | -0.87 [-0.94, -0.79] | -0.24 [-0.29, -0.18] | 6.2 [3.5, 9.3] | -0.65 [-0.87, -0.42] |
| sanoTTS heart | -1.26 [-1.35, -1.18] | -1.24 [-1.31, -1.17] | -0.20 [-0.25, -0.15] | 1.1 [0.0, 2.6] | -0.23 [-0.32, -0.15] |
| sanoTTS heart-nano | -1.88 [-1.96, -1.79] | -2.26 [-2.35, -2.17] | -0.37 [-0.43, -0.31] | 1.1 [0.0, 2.6] | -0.54 [-0.65, -0.44] |
| Inflect Nano v2 (Owen Song) | -0.13 [-0.23, -0.03] | -0.02 [-0.05, 0.01] | 0.06 [0.03, 0.08] | 0.6 [0.1, 1.1] | 0.98 [0.77, 1.18] |
| Inflect Micro v2 (Owen Song) | 0.25 [0.16, 0.33] | -0.02 [-0.06, 0.02] | 0.04 [0.01, 0.06] | 0.4 [0.0, 1.0] | 0.85 [0.65, 1.04] |
| TinyTTS (pschatzmann port) | -0.76 [-0.85, -0.66] | -0.77 [-0.84, -0.69] | -0.05 [-0.08, -0.02] | 6.2 [3.6, 9.4] | 0.62 [0.48, 0.75] |
| Piper lessac low | 0.18 [0.08, 0.29] | -0.02 [-0.05, 0.01] | 0.01 [-0.03, 0.05] | 0.7 [0.1, 1.5] | 1.55 [1.39, 1.70] |
| Piper lessac medium | 0.48 [0.40, 0.57] | -0.15 [-0.22, -0.09] | -0.06 [-0.12, 0.00] | 0.1 [-0.3, 0.6] | 1.62 [1.47, 1.77] |
| Piper amy low | 0.21 [0.11, 0.30] | 0.01 [-0.01, 0.03] | -0.07 [-0.12, -0.02] | 0.1 [-0.3, 0.6] | -1.04 [-1.17, -0.92] |
| Kitten TTS nano 0.8 | -1.22 [-1.33, -1.12] | -0.50 [-0.58, -0.42] | -0.03 [-0.07, -0.00] | 0.6 [-0.3, 1.6] | 1.36 [1.04, 1.68] |
| Kitten TTS micro 0.8 | -1.43 [-1.54, -1.31] | -0.46 [-0.52, -0.39] | -0.09 [-0.13, -0.06] | 0.3 [-0.6, 1.3] | 0.83 [0.57, 1.11] |
| Kitten TTS mini 0.8 | -1.33 [-1.42, -1.23] | -0.41 [-0.48, -0.34] | -0.07 [-0.10, -0.05] | 0.6 [-0.4, 2.0] | 0.41 [0.18, 0.63] |
| Kokoro-82M (af_heart) | 0.66 [0.58, 0.74] | 0.06 [0.05, 0.08] | 0.06 [0.03, 0.10] | 0.3 [-0.3, 1.0] | 0.06 [-0.08, 0.22] |
| MeloTTS EN-US | -0.18 [-0.28, -0.09] | -0.71 [-0.79, -0.63] | -0.33 [-0.40, -0.27] | 2.6 [1.2, 4.4] | 0.04 [-0.12, 0.22] |
| Pocket TTS (Kyutai, alba) | 0.10 [-0.01, 0.21] | -0.10 [-0.14, -0.06] | -0.03 [-0.07, 0.00] | 1.9 [0.7, 3.5] | 0.84 [0.58, 1.10] |
| Supertonic 2 (F1) | 0.41 [0.30, 0.51] | 0.01 [-0.02, 0.04] | 0.01 [-0.04, 0.04] | 1.8 [0.5, 3.5] | 0.00 [-0.29, 0.29] |
| Supertonic 3 (F1) | 0.63 [0.53, 0.74] | 0.02 [0.00, 0.05] | -0.02 [-0.06, 0.01] | 1.1 [0.3, 2.4] | -0.11 [-0.26, 0.04] |
| MOSS-TTS-Nano (ONNX, en_2 ref) | 0.22 [0.13, 0.32] | -0.06 [-0.11, -0.02] | -0.13 [-0.19, -0.07] | 1.1 [0.1, 2.4] | -0.54 [-0.70, -0.37] |
| eSpeak NG | -1.47 [-1.56, -1.38] | -2.29 [-2.37, -2.21] | -0.58 [-0.63, -0.54] | -0.3 [-0.8, 0.3] | -1.19 [-1.29, -1.10] |
| Flite (cmu_us_slt) | -1.66 [-1.73, -1.58] | -1.84 [-1.89, -1.79] | -0.75 [-0.80, -0.70] | 1.2 [0.3, 2.4] | -1.71 [-1.81, -1.63] |

Paired differences vs the **male** Ito row (system minus Ito male, same prompts; shown for the MCU-class and closest systems only, Mean [95% CI]; the other systems are single speakers, so this is the male voice against their voices):

| system | Δ UTMOSv2 | Δ UTMOS22 | Δ DNSMOS OVRL | Δ WER (pts) |
|---|---|---|---|---|
| Inflect Nano v2 (Owen Song) | -0.18 [-0.27, -0.08] | 0.00 [-0.03, 0.03] | -0.03 [-0.06, -0.00] | 0.4 [0.0, 0.9] |
| Inflect Micro v2 (Owen Song) | 0.20 [0.11, 0.29] | 0.00 [-0.04, 0.04] | -0.05 [-0.08, -0.02] | 0.3 [0.0, 0.7] |
| TinyTTS (pschatzmann port) | -0.80 [-0.92, -0.69] | -0.75 [-0.82, -0.68] | -0.13 [-0.17, -0.09] | 6.1 [3.5, 9.2] |
| sanoTTS amy | -0.45 [-0.55, -0.35] | -0.44 [-0.52, -0.37] | -0.25 [-0.31, -0.19] | 0.6 [-0.3, 1.7] |
| sanoTTS heart-nano | -1.92 [-2.01, -1.83] | -2.24 [-2.33, -2.14] | -0.45 [-0.53, -0.39] | 1.0 [-0.1, 2.3] |
| Kokoro-82M (af_heart) | 0.61 [0.53, 0.70] | 0.08 [0.07, 0.10] | -0.02 [-0.06, 0.01] | 0.1 [-0.5, 0.9] |
| StyleTTS 2 teacher (female voice D) | 0.18 [0.08, 0.28] | 0.06 [0.04, 0.07] | -0.09 [-0.14, -0.05] | 0.8 [-0.5, 3.2] |

| system | MACs per s of audio | weights | license | notes |
|---|---|---|---|---|
| **Ito 256 vocoder (first release) chip-exact (frontend fix)** (ours) | ~350 M int8 MACs executed (C engine count) | 4.89 MB blob | ours | same blob and engine; itofs_text.py normalize() spells out decimals / dollars / thousands before eSpeak (changes prompts 31 and 37 only) |
| **Ito 256 vocoder (first release) chip-exact** (ours, bench v1 frontend) | ~350 M int8 MACs executed (C engine count) | 4.89 MB blob | ours | the first-release acoustic front and 256-wide vocoder + fixed mean style; firmware C engine, host build, bit-identical to the chip; streaming, 100 ms chunks; text frontend WITHOUT normalize() (kept for transparency: '3.5 million' was cut to 'three') |
| **Ito 192 vocoder, chip-exact, female** (ours, shipped) | 263 M int8 MACs executed (C engine count) | 3.81 MB blob | ours | main int8 set, female voice; shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| **Ito 192 vocoder, chip-exact, male** (ours, shipped; no teacher row for this voice) | 263 M int8 MACs executed (C engine count) | 3.81 MB blob | ours | main int8 set, male voice (no teacher row); shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| Ito 192, int4 set, chip-exact, female | 263 M int8-equivalent MACs | 3.20 MB blob | ours | main int4 set (_int4 blob) female; shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| Ito 192, light set, chip-exact, female | ~247 M | 3.05 MB blob | ours | light set (one block fewer, int4) female; shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| Ito 192, int4 set, chip-exact, male | 263 M int8-equivalent MACs | 3.20 MB blob | ours | main int4 set male; shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| Ito 192, light set, chip-exact, male | ~247 M | 3.05 MB blob | ours | light set male; shipped 192/576 5-block vocoder; chip weight blob rendered by the firmware C engine (host build, bit-identical to the QEMU-verified chip PCM; verified here by equality with ito_cli at chunk 8 and 24), 100 ms host chunks, seed 1+prompt index, fixed mean style, text frontend with normalize(); scored on this Mac (Apple M4 Max, MPS) |
| Ito 256 vocoder (first release) float (ours) | ~322 M (logical) | 17.6 MB fp32 | ours | same checkpoints in PyTorch, streaming path, chunk 8 |
| StyleTTS 2 LibriTTS, voice D (our teacher) | n/p | ~760 MB | MIT code; LibriTTS-R speaker 4970 | diffusion style, 5 steps, alpha 0.3 / beta 0.7, mean ref style of 5 clips; the model Ito was distilled from |
| sanoTTS amy | n/p | 2.9 MB fp16 | runtime MIT; project GPLv3 | distilled from Piper en_US-amy-medium, 22.05 kHz |
| sanoTTS kristin | n/p | 2.8 MB fp16 | runtime MIT; project GPLv3 | distilled from Piper en_US-kristin-medium |
| sanoTTS heart | n/p | 9.1 MB fp32 | runtime MIT; project GPLv3 | distilled from Kokoro af_heart, 24 kHz |
| sanoTTS heart-nano | n/p (567 K voice: ~45 M, paper) | 337 KB | runtime MIT; project GPLv3 | int8, 24 kHz |
| Inflect Nano v2 (Owen Song) | n/p | 16.0 MB fp32 | Apache-2.0 model | VITS-family, fixed male voice, 24 kHz; package defaults |
| Inflect Micro v2 (Owen Song) | n/p | 37.5 MB fp32 | Apache-2.0 model | fixed male voice, 24 kHz; package defaults |
| TinyTTS (pschatzmann port) | n/p | 4.2 MB | Apache-2.0 | C++ port of tronghieuit/tiny-tts (VITS); desktop CLI = same code as the MCU build, defaults |
| Piper lessac low | n/p | 63 MB | MIT code; voice MODEL_CARD | VITS, 16 kHz |
| Piper lessac medium | n/p | 63 MB | MIT code; voice MODEL_CARD | VITS, 22.05 kHz |
| Piper amy low | n/p | 63 MB | MIT code; voice MODEL_CARD | VITS, 16 kHz |
| Kitten TTS nano 0.8 | n/p | 57 MB ONNX | Apache-2.0 | StyleTTS 2-based, voice Bella |
| Kitten TTS micro 0.8 | n/p | 41 MB ONNX | Apache-2.0 | voice Bella |
| Kitten TTS mini 0.8 | n/p | 78 MB ONNX | Apache-2.0 | voice Bella |
| Kokoro-82M (af_heart) | n/p | 327 MB | Apache-2.0 | StyleTTS 2 + iSTFTNet, 24 kHz |
| MeloTTS EN-US | n/p | ~200 MB | MIT | VITS2-based, 44.1 kHz |
| Pocket TTS (Kyutai, alba) | n/p | ~440 MB | MIT code; CC-BY-4.0 weights | streaming continuous-latent LM, 24 kHz |
| Supertonic 2 (F1) | n/p | ~260 MB ONNX | MIT code; OpenRAIL-M model | flow matching, 8 steps, 44.1 kHz |
| Supertonic 3 (F1) | n/p | ~400 MB ONNX | MIT code; OpenRAIL-M model | flow matching, 8 steps, 44.1 kHz |
| MOSS-TTS-Nano (ONNX, en_2 ref) | n/p | ? | Apache-2.0 | AR LM + audio tokenizer, voice cloned from repo sample en_2, 48 kHz |
| eSpeak NG | – | ~2 MB data (en) | GPLv3 | formant synthesis, en-us |
| Flite (cmu_us_slt) | – | ~5 MB | BSD-style | statistical parametric |

Mean clip duration (s) / DNSMOS SIG / BAK / F0 range p5–p95 (st):

| system | dur | SIG | BAK | F0 range |
|---|---|---|---|---|
| **Ito 256 vocoder (first release) chip-exact (frontend fix)** (ours) | 5.08 [4.56, 5.63] | 3.62 [3.60, 3.65] | 4.16 [4.15, 4.17] | 9.09 [8.85, 9.32] |
| **Ito 256 vocoder (first release) chip-exact** (ours, bench v1 frontend) | 5.02 [4.49, 5.59] | 3.62 [3.59, 3.65] | 4.16 [4.14, 4.17] | 9.10 [8.86, 9.33] |
| **Ito 192 vocoder, chip-exact, female** (ours, shipped) | 5.09 [4.56, 5.63] | 3.59 [3.57, 3.61] | 4.13 [4.11, 4.15] | 9.05 [8.83, 9.28] |
| **Ito 192 vocoder, chip-exact, male** (ours, shipped; no teacher row for this voice) | 5.34 [4.85, 5.87] | 3.67 [3.66, 3.69] | 4.16 [4.14, 4.17] | 7.86 [7.60, 8.12] |
| Ito 192, int4 set, chip-exact, female | 5.08 [4.56, 5.63] | 3.59 [3.57, 3.61] | 4.13 [4.11, 4.15] | 9.09 [8.86, 9.33] |
| Ito 192, light set, chip-exact, female | 5.12 [4.60, 5.67] | 3.61 [3.58, 3.64] | 4.15 [4.14, 4.17] | 9.05 [8.80, 9.29] |
| Ito 192, int4 set, chip-exact, male | 5.34 [4.85, 5.87] | 3.67 [3.65, 3.69] | 4.16 [4.14, 4.17] | 7.86 [7.60, 8.10] |
| Ito 192, light set, chip-exact, male | 5.34 [4.84, 5.86] | 3.64 [3.63, 3.66] | 4.13 [4.11, 4.14] | 8.05 [7.62, 8.65] |
| Ito 256 vocoder (first release) float (ours) | 5.03 [4.49, 5.59] | 3.63 [3.60, 3.65] | 4.17 [4.16, 4.18] | 9.53 [8.99, 10.24] |
| StyleTTS 2 LibriTTS, voice D (our teacher) | 5.25 [4.66, 5.88] | 3.58 [3.55, 3.61] | 4.12 [4.08, 4.16] | 9.23 [8.88, 9.59] |
| sanoTTS amy | 5.08 [4.48, 5.72] | 3.53 [3.49, 3.56] | 3.92 [3.86, 3.98] | 7.21 [5.98, 8.52] |
| sanoTTS kristin | 4.70 [4.14, 5.30] | 3.47 [3.44, 3.50] | 3.87 [3.80, 3.93] | 7.16 [6.35, 8.01] |
| sanoTTS heart | 4.30 [3.75, 4.89] | 3.41 [3.34, 3.47] | 4.02 [3.99, 4.05] | 8.27 [8.02, 8.52] |
| sanoTTS heart-nano | 4.32 [3.78, 4.90] | 3.28 [3.20, 3.35] | 3.92 [3.85, 3.97] | 7.46 [7.25, 7.66] |
| Inflect Nano v2 (Owen Song) | 4.36 [3.86, 4.89] | 3.61 [3.59, 3.63] | 4.16 [4.16, 4.17] | 12.66 [11.94, 13.34] |
| Inflect Micro v2 (Owen Song) | 4.35 [3.85, 4.89] | 3.60 [3.57, 3.62] | 4.15 [4.14, 4.16] | 12.26 [11.48, 13.04] |
| TinyTTS (pschatzmann port) | 5.91 [5.23, 6.66] | 3.55 [3.53, 3.58] | 4.09 [4.06, 4.11] | 11.29 [10.89, 11.71] |
| Piper lessac low | 4.30 [3.79, 4.85] | 3.62 [3.59, 3.64] | 4.10 [4.06, 4.14] | 14.14 [13.53, 14.71] |
| Piper lessac medium | 4.20 [3.70, 4.73] | 3.59 [3.56, 3.62] | 4.02 [3.96, 4.08] | 14.41 [13.89, 14.92] |
| Piper amy low | 4.83 [4.31, 5.39] | 3.60 [3.58, 3.62] | 3.98 [3.91, 4.04] | 6.00 [5.69, 6.30] |
| Kitten TTS nano 0.8 | 7.44 [6.69, 8.26] | 3.53 [3.51, 3.55] | 4.17 [4.16, 4.18] | 14.32 [13.00, 15.67] |
| Kitten TTS micro 0.8 | 7.92 [7.16, 8.72] | 3.48 [3.45, 3.51] | 4.14 [4.11, 4.15] | 11.39 [10.32, 12.50] |
| Kitten TTS mini 0.8 | 6.58 [5.90, 7.30] | 3.51 [3.48, 3.53] | 4.13 [4.11, 4.15] | 10.08 [9.24, 11.04] |
| Kokoro-82M (af_heart) | 4.30 [3.76, 4.89] | 3.64 [3.62, 3.66] | 4.15 [4.12, 4.16] | 9.29 [8.87, 9.94] |
| MeloTTS EN-US | 4.58 [4.05, 5.14] | 3.51 [3.48, 3.53] | 3.64 [3.54, 3.73] | 9.29 [8.75, 9.88] |
| Pocket TTS (Kyutai, alba) | 3.83 [3.38, 4.30] | 3.56 [3.52, 3.60] | 4.10 [4.08, 4.12] | 12.19 [11.27, 13.13] |
| Supertonic 2 (F1) | 5.00 [4.42, 5.63] | 3.59 [3.54, 3.63] | 4.14 [4.10, 4.16] | 9.06 [8.08, 10.11] |
| Supertonic 3 (F1) | 4.25 [3.69, 4.85] | 3.59 [3.56, 3.61] | 4.10 [4.07, 4.13] | 8.66 [8.28, 9.00] |
| MOSS-TTS-Nano (ONNX, en_2 ref) | 5.11 [4.55, 5.72] | 3.55 [3.52, 3.58] | 3.97 [3.90, 4.03] | 7.46 [7.02, 7.92] |
| eSpeak NG | 4.24 [3.75, 4.76] | 3.19 [3.15, 3.22] | 3.65 [3.58, 3.71] | 5.34 [5.08, 5.59] |
| Flite (cmu_us_slt) | 4.52 [4.00, 5.07] | 2.91 [2.85, 2.96] | 3.83 [3.78, 3.88] | 3.53 [3.41, 3.64] |

## Method

- **Rendering** (`synth_*.py`; v1.1 rows: `synth_inflect.py`, `synth_tinytts.py`, `synth_pocket.py`,
  `synth_supertonic.py`, `synth_moss.py`, envs from `tools/setup_envs_small.sh`, raw output in `raw/<system>/`, never committed): Ito chip = `synth_ito_chip.py`
  (the released `ito_{female,male}_esp32s3.bin` blobs and the engine in `esp32/engine`, 8-bit activations, 100 ms host chunks (output is identical for other chunk sizes), style 0 = the blob's fixed mean style, seed
  1 + index); Ito float = PyTorch streaming path (chunk 8); teacher = StyleTTS2-LibriTTS with speaker 4970's mean
  reference style (5 clips), 5 diffusion steps, alpha 0.3 / beta 0.7; Ito and the teacher read the same phonemes
  (`tools/ito_tokens.py`, outputs in `tools/ito_frontend_out/`). Every baseline uses its own frontend and default
  settings: sanoTTS 0.5.0 pip runtime (GPLv3 project, separate venv, benchmark use only, no code copied); Piper
  (piper-tts, voices v1.0.0); Kitten TTS 0.8.1 (voice Bella); Kokoro 0.9.4 (af_heart); MeloTTS (EN-US); eSpeak NG
  (git master, en-us); Flite (cmu_us_slt); Inflect v2 HF packages (inference.py defaults, seed = prompt index);
  TinyTTS desktop CLI defaults; Pocket TTS voice alba; Supertonic F1 (8 steps); MOSS-TTS-Nano public ONNX runtime cloning
  its own English sample en_2 (48 kHz stereo averaged to mono). SVOX Pico was attempted, but our 64-bit build returned empty audio
  nondeterministically, so it is excluded.
- **Post-processing** (`score.py`, identical for all): mono, 24 kHz (soxr), silence trim (top_db 40 + 50 ms pad),
  -20 LUFS, peak -1 dBFS, 16-bit. Showcase prompts (12 per system) are in `audio/<system>/`; `audio/ito_v3_chip/chip_pcm/`
  holds the untouched chip PCM for the same prompts (first-release rows only; the 192 rows' raw PCM is regenerated by `synth_ito_chip.py`).
- **Metrics:** UTMOSv2 (sarulab-speech, pretrained); UTMOS22 strong (SpeechMOS v1.2.0); DNSMOS P.835 (sig_bak_ovr.onnx);
  Whisper large-v3 (transformers fp16, beam 5) WER/CER with Whisper's EnglishTextNormalizer plus two rules on both sides
  ("3.30"/"3:30" → "3 30"; interjection spellings hm/hmm/hem dropped); pYIN F0 (50–600 Hz). Per-clip values, transcripts
  and normalised texts are in `results_per_clip.json`.
- **Statistics:** mean over the 54 prompts, 95% percentile bootstrap (10,000 prompt resamples); WER/CER are corpus-level
  ratios bootstrapped the same way; paired deltas vs Ito chip resample prompts jointly.
- **Reproduce:** `bash run_all.sh` (see its header for the env vars; scoring needs one GPU, or `BENCH_DEVICE=mps|cpu`). The baselines and the first-release Ito rows were scored on EC2 g5.4xlarge
  (A10G) boxes (CUDA fp16 Whisper), CPU rendering for all baselines. **The 192-wide Ito rows (2026-10-08) were scored on an Apple M4 Max (MPS; Whisper large-v3 fp16 beam 5, UTMOSv2/UTMOS22/DNSMOS/pYIN as above) with the same code and the stored per-clip values of every other system.** Cross-check of the Mac setup: re-scoring 12 stored clips of the first-release row and 12 of the teacher reproduced UTMOS22 (mean abs. difference 0.0001-0.0002), DNSMOS (0.001), F0 SD (0.0001) and all 24 transcripts exactly. **UTMOSv2 is stochastic** (it draws random crops: a clip's score moves by about 0.25 between calls, a 54-clip mean by about 0.03), so a re-score of the same audio does not reproduce a UTMOSv2 mean exactly; the same holds for every row's stored value. The Mac run seeds it (`random.seed(0)`, `torch.manual_seed(0)`) and uses one draw per clip, like the published rows.
- **Listening test:** `python listening_test/make_test.py --out site/ --mode acr|mushra [--systems ...] [--submit_url ...]`
  builds a static page with hidden, randomised labels (the key is written next to `site/`, never inside it);
  `listening_test/analyze.py key_*.json ratings_*.json` gives per-system MOS with listener×prompt bootstrap CIs.

