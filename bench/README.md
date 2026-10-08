# Benchmark

Ito against its teacher and 21 public small and edge English TTS systems, on 54 fixed English prompts (`prompts.json`) that do not occur in Ito's training text. Every system's audio goes through the same post-processing (mono, 24 kHz, silence trim, -20 LUFS, peak -1 dBFS) and is scored with UTMOSv2, UTMOS22, DNSMOS P.835 and Whisper large-v3 (word and character error rate), with 95 % bootstrap intervals over prompts and paired differences between systems.

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

- **Ito rows** are the shipped model (192-wide vocoder, released weight blobs, main int8 set) rendered by the host build of the on-chip engine, whose PCM is bit-identical to the QEMU-verified firmware; female and male voice, plus the int4 and light weight sets (see `results.md`). The male voice has no teacher row. The 256-wide row is the first release.
- **Scoring machine.** The Ito 192 rows were scored on 8 October 2026 on an Apple M4 Max (`BENCH_DEVICE=mps python score.py ...`); every other system's values are the stored ones from the earlier GPU run. Re-scoring 24 stored clips on the Mac reproduced UTMOS22, DNSMOS, F0 and all transcripts. **UTMOSv2 is stochastic** (random crops: about 0.25 per clip, about 0.03 on a 54-clip mean), so its small differences are within noise.
- Speed columns are not like-for-like: Ito's time to first audio and real-time factor were measured on one board for the female main set (first audio chunk computed in 184 ms, excluding the DAC stage; RTF 0.66; `esp32/README.md` §4a) and are pre-board estimates for the rest; Inflect Nano's and TinyTTS's figures are from their own READMEs.
- Files: `results.md` (all tables, paired differences, findings, method), `results.json` and `scatter.json` (numbers), `results_per_clip.json` (per-clip scores and transcripts), `prompts.json`, `score.py` (the scoring script). The renderers for the other systems and the listening-test tools are not included.
