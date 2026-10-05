# Results

## blind10/: listening test #10 (5 October 2026)

`ratings.json`: the lighter vocoder. One expert listener rated 16 clips (4 systems x the same 4 sentences as #9) of the female voice for
naturalness, 1-5, system names hidden: teacher 4.25, Ito with the 256-wide vocoder 4.00, Ito with the 192-wide vocoder 4.00, Ito with the
192-wide vocoder and int4 weights 4.00. He heard no difference between the three Ito systems. One listener, four clips per system.


## blind9/: listening test #9 (2 October 2026)

- `ratings.json`: per-clip scores, the unblinded key, and the per-system means. One expert listener rated 24 clips
  (6 systems × 4 sentences) for naturalness on a 1–5 scale, with the system names hidden.
- `sentences.json`: the eight test sentences. Sentences 01, 03, 04 and 06 were rated.
- `metrics.json`: UTMOS and Whisper WER for all systems on all eight sentences (medium.en and base, after number
  normalisation).
- `audio/ito/`, `audio/teacher/`: the rated clips of Ito and of the teacher, for all eight sentences. The Ito clips
  come from the float PyTorch model, through the streaming path, with the optional text-to-style predictor. All clips
  are trimmed and normalised to −20 LUFS. The sanoTTS baseline clips are not redistributed.

| system | mean | per sentence (01 / 03 / 04 / 06) |
|---|---|---|
| teacher (StyleTTS 2 LibriTTS model, voice of LibriTTS-R speaker 4970) | 4.75 | 5 / 5 / 4 / 5 |
| **Ito, streaming front (released)** | **4.00** | 4 / 4 / 4 / 4 |
| Ito, bidirectional front | 4.00 | 4 / 4 / 4 / 4 |
| Ito, 384-wide vocoder | 3.75 | 4 / 4 / 4 / 3 |
| sanoTTS amy | 2.00 | 2 / 3 / 2 / 1 |
| sanoTTS heart-nano | 1.00 | 1 / 1 / 1 / 1 |

There was one listener and four sentences per system, so differences under about half a point are not meaningful.

## blind8/: listening test #8 (1 October 2026)

Two earlier Ito variants against the reference (teacher) model; one listener, four held-out sentences. Reference
4.00, variant A 3.25, variant B 3.00.

## chip/: the on-chip engine (`ito_female_esp32s3.bin`, `ito_male_esp32s3.bin`; the 192-wide vocoder of 5 October)

- `quality.json`, `quality_c.log`: quantised engine numerics against the float PyTorch model on 20 held-out sentences
  (PESQ-wb 4.50 female / 4.55 male on 10 sentences; durations identical on 10/10; `_male` files for the male voice), and the C engine's audio against both.
- `host_test_a8.log`, `host_test_a16.log`: the C engine against the engine-numerics reference, stage by stage, with
  8- and 16-bit activations. Streaming is bit-identical to whole-utterance output.
- `qemu_rtf_female.log`, `qemu_rtf_male.log`: the firmware in QEMU with the 192-wide blobs, all four weight-staging modes, PCM bit-identical to the host. `qemu_icprof_*.log.xz` and `estimates_5oct.txt`: exact instruction counts and the resulting estimates (`estimates_3oct.txt`: the 256-wide vocoder).
- `qemu_sets_*.log`, `qemu_modes_*.log`, `host_test_sched.log`, `int4/`: the weight sets (main int8, main int4, light) and the boot self-calibration: the firmware in QEMU with all three sets of a voice in flash (selection with injected timings, PCM of every set bit-identical in all staging modes, C and PIE int4 unpack), the host test of the start-delay policy, and the int4 exactness checks and quality logs.
- `qemu_v3.log`: an earlier run of the firmware in Espressif QEMU 9.2.2, built from this repository. The self-test passes, and its PCM is
  bit-identical to the host build (1 and 2 cores, 8- and 16-bit activations, two weight-staging modes). Its timing
  lines are emulator wall-clock times, not chip times.
- `opcount.log`, `estimates.md`: exact operation counts and the **estimated** time to first audio and real-time factor
  as a function of the board's effective int8 throughput. These are not silicon measurements, and `estimates.md` is superseded by the instruction-count estimates in `esp32/README.md` section 4 (it was too optimistic).

## License of the audio

The Ito audio here (and in `samples/`) is output of the Ito voice model and is shared under CC BY-NC-SA 4.0, with
Lokutor's terms in [`models/TERMS.md`](../models/TERMS.md).
