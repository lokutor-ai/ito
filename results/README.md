# Results

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

## chip/: the on-chip engine (`ito_female_esp32s3.bin`)

- `quality.json`, `quality_c.log`: quantised engine numerics against the float PyTorch model on 20 held-out sentences
  (PESQ-wb 4.52; durations identical on 20/20), and the C engine's audio against both.
- `host_test_a8.log`, `host_test_a16.log`: the C engine against the engine-numerics reference, stage by stage, with
  8- and 16-bit activations. Streaming is bit-identical to whole-utterance output.
- `qemu_v3.log`: the firmware in Espressif QEMU 9.2.2, built from this repository. The self-test passes, and its PCM is
  bit-identical to the host build (1 and 2 cores, 8- and 16-bit activations, two weight-staging modes). Its timing
  lines are emulator wall-clock times, not chip times.
- `opcount.log`, `estimates.md`: exact operation counts and the **estimated** time to first audio and real-time factor
  as a function of the board's effective int8 throughput. These are not silicon measurements, and `estimates.md` is superseded by the instruction-count estimates in `esp32/README.md` section 4 (it was too optimistic).

## License of the audio

The Ito audio here (and in `samples/`) is output of the Ito voice model and is shared under CC BY-NC-SA 4.0, with
Lokutor's terms in [`models/TERMS.md`](../models/TERMS.md).
