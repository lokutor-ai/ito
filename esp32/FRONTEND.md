# On-chip text front end (text -> Ito token ids, no computer attached)

`esp32/frontend/` turns English text into the token ids the host front end (`ito/text.py`, `tools/say.py`) produces, on the ESP32-S3 itself.
Firmware commands: `text <string>` (speak), `textq <string>` (compute only, no I2S, prints timing and PCM hash), `ids <string>` (front end only,
prints the ids), `phrases [q]` (the phrase list stored in flash, `firmware/main/phrases.txt`), `fastfirst 0|1`. Build options (menuconfig `Ito`):
`ITOFS_FRONTEND` (default y), `ITOFS_FE_FAST_FIRST` (y), `ITOFS_BOOT_TEXT` (speak `phrases.txt` at boot, like `ITOFS_BOOT_DEMOS`), `ITOFS_BOOT_TEXT_SILENT`
(boot phrases are computed and hashed, not played). Prebuilt image: `prebuilt/ito_app_frontend_merged.bin` (flash at 0x0, boots silent as before;
needs the usual weight partitions).

> **Status of the measurements.** Correctness (ids) was verified on the host and in Espressif QEMU running the real firmware. **The physical board
> could not be reached in this session** (the CH343 UART port enumerates but neither esptool nor a plain reset gets any byte back; it needs BOOT+RST
> pressed by hand). So there is **no on-silicon timing or memory number for the front end yet**: the latency below is QEMU instruction counts
> converted with an assumed CPI, and is labelled as an estimate. The command `ids` / `textq` and `tools/serial_ids.py` are ready to measure it.

## Approach and why

| | A: port of espeak-ng 1.52.0 (en-us) | B: CMUdict-class dictionary + rules + normalizer | C: int8 neural G2P |
|---|---|---|---|
| Reference | the host uses espeak-ng 1.52.0 (espeakng-loader 0.2.4), so the *same code and data* reproduce it | would imitate espeak's phonemes, stress marks, number/abbreviation expansion, "tuːhʌndɹɪd" fusions, ˌ/ˈ placement: no exact match possible | would imitate it with errors |
| Result | **identical ids on 100 % of 27,000+ lines (below)** | not built | not built (A succeeded) |

B and C were not built because A reached exact agreement; B/C cannot match a rule engine's quirks id for id, and the host data is espeak-ng's.
What the port contains: the translator only (text -> phonemes), no synthesis. Files kept: `translate, translateword, dictionary, numbers, readclause,
phonemelist, synthdata, voices (LoadVoice only), tr_languages, langopts, mnemonics, encoding, setlengths, common, ssml, error` + ucd-tools; stubs for the
synthesis symbols (`ito_stubs.c`); the data directory is replaced by one memory blob (`ito_vfs.c`), read in place from flash with no copy.
All edits to espeak-ng files are marked `ITO` and listed in `frontend/espeak/PORT.md`. Licence: espeak-ng GPLv3+ (`espeak/COPYING`), kept; the rest GPLv3+.
Data blob `frontend/data/espeak_en_us.bin` (273 KB): phontab, phonindex, intonations, en_dict, voice `gmw/en-US`, and only the 8-byte header of
phondata (the translator never reads the waveform part: output is identical with it removed, checked on all test sets).

On top of espeak, `ito_frontend.c` reproduces the rest of the host pipeline: `normalize()` (commas, `$`, decimals, Unicode digits), `strip`/`"` removal, the
phonemizer 3.x espeak backend (punctuation split and restore, inline quote/bracket groups phonemized in context, `_postprocess_line`), a subset of nltk's
`word_tokenize` (Punkt sentence breaks including its abbreviation list and "initial" rule, then the Treebank rules for the characters a phoneme string can
contain), the StyleTTS 2 symbol table, and `tools/say.py`'s splitting (sentences at `[.!?;]`+space, middle-word split above 400 tokens, merge up to 200 tokens).

## Agreement with the host front end (exact id sequence, per line)

Reference = `ito.text.text_to_ids` with phonemizer 3.6.0, nltk 3.10.4, espeak-ng 1.52.0 (`tools/make_reference.py`; it refuses to run if nltk is missing,
because `ito/text.py` silently falls back to a regex tokenizer then). Test sets: `corpus/corpus.tsv` (1403 sentences: prose, the 3 demos, numbers, dates, times,
currency, units, abbreviations, acronyms, punctuation, contractions, names, misc, long), `corpus/edge.tsv`, `corpus/nonenglish.tsv`, and generated stress sets.

| set | lines | identical ids |
|---|---|---|
| corpus (14 categories, each 100 %) | 1403 | 1403 (100.00 %) |
| random words/abbreviations/numbers/punctuation/odd whitespace (3 seeds) | 18,000 | 18,000 |
| hostile Unicode (accents, Greek, Cyrillic, CJK, Arabic digits, emoji, combining marks, NBSP, ZWSP) | 4,000 | 4,000 |
| edge cases + non-English | 38 | 36; the 2 are single "sentences" > 1000 tokens (see limits) |
| say.py utterance lists, multi-sentence texts (1-9 sentences, run-ons) | 412 (899 utterances) | 408 texts; the 4 differ only where the host's utterance exceeds 400 tokens (the chip cannot speak those; we truncate) |
| same, **on the chip firmware in QEMU** (`tools/qemu_run.sh`): corpus + multi + 2500 fuzz | 4,315 | 4,315 |

Vs. an older phonemizer (<= 3.3.0 punctuation logic, emulated by `--legacy-punct`): 1401 / 1403; the 2 differ by one token each ("bᵻlˌoʊ" vs "bᵻlˈoʊ" stress
mark; list bullet "a)" read as "ɐ" vs "ˈeɪ"). That is a difference between host phonemizer versions, not a port error; I did not synthesize those two (no on-chip
mismatch exists to compare).
Honest finding: the **stored demo 2 token array** in `main/demo_sentences.h` differs from what espeak-ng 1.52.0 (and therefore the host front end today)
gives in two phonemes ("ɪŋ" vs "ɪn" in "in consequence", "ɪ" vs "ᵻ" in "curiosity"): it was made with an older espeak-ng. Demos 0 and 1 are identical.
`make test_frontend` (esp32/host) asserts exactly that.

## Memory

| | flash | PSRAM | internal SRAM |
|---|---|---|---|
| data blob | 273 KB (in the app image, read through the cache) | - | - |
| code + tables | 129 KB code + 165 KB const tables | - | - |
| static data (.bss 68 KB, placed in PSRAM by `linker.lf` + `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY`) | - | 68 KB | **0** (.data: 76 bytes) |
| working buffers (`itofe_work_bytes`) + stream state + text/req buffers | - | 216 + ~13 KB | 0 |
| task stack (64 KB allocated, **peak 45.8 KB** measured; espeak keeps clause buffers on the stack) | - | 64 KB | 0 (TCB ~0.4 KB) |
| espeak heap (translator, voice, decoder) | - | all allocations are redirected to PSRAM (`ito_mem.h`) | 0 |
| **total** | **~570 KB** (app 411 KB -> 996 KB of the 2 MB slot; partition table unchanged) | **~0.36 MB** (QEMU: peak after the self-test 5674 KB vs 5372 KB on the board without it) | **~0** |

The partition table was **not changed**: the data lives in the app image. The internal-SRAM high-water mark of the engine (354 of 374 KB) is untouched in the
static link (DIRAM +76 bytes); the run-time figure on silicon is not measured (no board access). A first version with a 40 KB stack overflowed it silently
into a neighbouring heap block (found with the QEMU heap checker); 64 KB is the shipped value.

## Latency (estimate, not silicon)

QEMU `-icount shift=0` counts exact instructions: over the 1403-sentence corpus, front end per utterance: median 0.80 M instructions (70 ids), mean 1.03 M,
p90 1.6 M, p99 4.4 M, max 7.1 M; fit `0.10 M + 0.0117 M x ids` (60 ids: 0.8 M, 100: 1.26 M, 200: 2.4 M). Init is a one-off, 9 ms in QEMU. At 240 MHz:

| utterance | instructions | at CPI 1.5 | at CPI 3 (pessimistic: flash/PSRAM cache misses) |
|---|---|---|---|
| 60 ids (a short sentence) | 0.8 M | 5 ms | 10 ms |
| 100 ids | 1.26 M | 7.9 ms | 15.8 ms |
| 200 ids (the grouping limit) | 2.4 M | 15 ms | 30 ms |

So the <= 30 ms target for the first utterance is expected to hold for every sentence up to ~200 tokens even pessimistically, **but this is an estimate**. CPI is
an assumption. Policy: with `ITOFS_FE_FAST_FIRST` only the first sentence is phonemized before the engine starts (first-chunk latency of the engine stays 184 ms,
so text -> first audio ~ 190-200 ms); the following sentences are phonemized by the front-end task (priority 3, below the synthesis) while the engine waits for
the player, so RTF 0.66 is not touched (not measured on silicon). `fastfirst 0` gives exactly say.py's utterances (a second sentence is phonemized first).
Not implemented: splitting a very long first sentence at its first comma (that would change the ids from the host's).

## Limits and what fails

- English (en-us) only. Foreign words get espeak's English guess; scripts that make espeak switch voice are not switched (matched the host on all tests, but not guaranteed).
- One utterance <= 2400 bytes after normalization and <= 400 tokens for the engine; the stream splitter handles long text at sentence ends and mid-word otherwise
  (say.py's rule); a *single* sentence over ~600 characters without any `.!?;` can give a first half over 400 tokens, which the engine cannot take (truncated).
- The tokenizer is a hand-written subset of nltk: exact on every test, but characters/patterns not in the tests could differ (e.g. nltk contraction rules for all-ASCII words).
- Matches *this* phonemizer/nltk/espeak-ng trio; the host result changes with those versions (see above).
- The `text` command takes one line (<= 4 KB) of UTF-8 over the console.

## Reproduce

```
cd esp32/host && make test_frontend                   # C front end vs golden ids from Python (no weights, no Python)
cd esp32/frontend && make && python3 tools/make_reference.py corpus/corpus.tsv ref.jsonl && build/fe_cli data/espeak_en_us.bin corpus/corpus.tsv > c.jsonl \
  && python3 tools/diff_corpus.py ref.jsonl c.jsonl      # needs phonemizer, espeakng-loader, nltk (+punkt_tab)
tools/qemu_run.sh <fw build dir> models/ito_female_esp32s3.bin ref.jsonl --stream --fastfirst0   # real firmware in QEMU
tools/serial_ids.py --port /dev/cu.usbserial-XXXX ref.jsonl     # same on a board (`ids <text>`), also prints the chip's front-end ms
```
