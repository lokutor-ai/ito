# On-chip text front end (text -> Ito token ids, no computer attached)

`esp32/frontend/` turns English text into the token ids the host front end (`ito/text.py`, `tools/say.py`) produces, on the ESP32-S3 itself.
Firmware commands: `text <string>` (speak), `textq <string>` (compute only, no I2S, prints timing and PCM hash), `ids <string>` (front end only,
prints the ids), `textp <string>` (like `textq`, with an emulated playout clock: reports the gap to the previous utterance and underruns), `phrases [q]` (the phrase list stored in flash, `firmware/main/phrases.txt`), `fastfirst 0|1`. Build options (menuconfig `Ito`):
`ITOFS_FRONTEND` (default y), `ITOFS_FE_FAST_FIRST` (y), `ITOFS_BOOT_TEXT` (speak `phrases.txt` at boot, like `ITOFS_BOOT_DEMOS`), `ITOFS_BOOT_TEXT_SILENT`
(boot phrases are computed and hashed, not played). Prebuilt image: `prebuilt/ito_app_frontend_merged.bin` (the default prebuilt since v1.1.0; `tools/flash.sh` writes it; flash at 0x0, boots silent as before;
needs the usual weight partitions; sha256 `9ef9553900aaaeadb522dfe41fc2816aecf2ac41cf9c2a35eb76a40c41f19eff`, ESP-IDF v5.5.1, built twice to the same bytes). `ITOFS_AUTOTEST` builds a stand-alone test image (see "Stand-alone run").

> **Status of the measurements.** Correctness (ids) was verified on the host and in Espressif QEMU, **and since 5 November 2026 on silicon**: every number in the
> "On silicon" sections below was measured on one board (ESP32-S3 QFN56 rev v0.2, MAC f8:5b:1b:91:64:ac, 8 MB octal PSRAM, 16 MB flash; firmware = the commits named below, now on `main`,
> ESP-IDF v5.5.1, female voice; the commands are in `frontend/tools/` and the "Reproduce" section below; the raw logs are kept by the authors, not in this repository).
> One board (not the board of the earlier measurements in `README.md`), one chip revision, USB powered; not a second board for the front end, not a cold power-up from an unplugged state (a physical unplug was impossible: see
> "Stand-alone run" below for what was done instead and what it proves).

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

## Memory (link figures, then silicon)

| | flash | PSRAM | internal SRAM |
|---|---|---|---|
| data blob | 273 KB (in the app image, read through the cache) | - | - |
| code + tables | 129 KB code + 165 KB const tables | - | - |
| static data (.bss 68 KB, placed in PSRAM by `linker.lf` + `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY`) | - | 68 KB | **0** (.data: 76 bytes) |
| working buffers (`itofe_work_bytes`) + stream state + text/req buffers | - | 216 + ~13 KB | 0 |
| task stack (64 KB allocated, **peak 45.8 KB** on silicon; espeak keeps clause buffers on the stack) | - | 64 KB | 0 (TCB ~0.4 KB) |
| espeak heap (translator, voice, decoder) | - | all allocations are redirected to PSRAM (`ito_mem.h`) | 0 |
| **total** | **~570 KB** (app image 948 KB of the 2 MB slot) | **~0.36 MB** | **~0** |

**On silicon** (board and commit under "Measured on silicon" below): after boot with the front end up and the female int4 set loaded, free PSRAM 2143 KB of 8125 KB and free internal SRAM 52 KB, **identical before and after 579 `textq`
utterances (500 texts), 2380 `ids` utterances and 1200 fuzz inputs** (no leak); high-water marks: PSRAM 5982 KB (engine + front end + the 12 s text PCM ring; 6251 KB while the 3.8 MB int8 set is loaded), internal SRAM 346 of 371 KB (unchanged by
the front end: the engine's hot arena stays in internal SRAM), minimum free internal SRAM 25 KB, front-end stack 45.8 of 64 KB. A first version with a 40 KB stack overflowed it silently into a neighbouring heap block (found with the QEMU heap checker); 64 KB is the shipped value.
The build matters: with an old `build_hw/sdkconfig` that lacks `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY` the front end's static data takes internal SRAM, the hot arena falls back to PSRAM and the engine is ~25 % slower;
`tools/build_fw.sh` now builds from a clean directory, and the boot log must say `arena: hot 210764 bytes in internal SRAM`.

## Measured on silicon (5 November 2026)

Board: ESP32-S3 QFN56 rev v0.2, MAC f8:5b:1b:91:64:ac, 8 MB octal PSRAM, 16 MB flash (Boya), USB powered; firmware: commits 61027d5, b343656, 08e8bae and the documentation commit e55e205 (now on `main`), ESP-IDF v5.5.1, female voice,
full split (3 s supply test: 272-273 passes in 3.02 s, pass; bit-exact self-test passes in every boot), boot-chosen int4 set unless stated. Commands: "Reproduce" below; raw logs are kept by the authors.
No sound was produced: `textq`, `textp` and the stand-alone mode never start I2S.

**Correctness**: the chip's utterance ids equal the Python front end's (phonemizer 3.6.0, nltk 3.10.4, espeak-ng 1.52.0) for **2004 of 2004 texts** (1403 corpus + 601 fresh texts written for this run: URLs, e-mail, numbers, dates, typos, names, mixed punctuation), 527 of 527
in say.py-exact mode, and the chip's PCM equals the host C engine's for **all 1,661 utterances compared** (154 + 748 + 180 + 579).

**Front end alone** (`ids`, 2380 utterances): median 6.8 ms, mean 8.4, p95 26.5, p99 36.3, max 58.5; by length: 0-49 ids median 3.5 ms, 50-99 7.5, 100-149 10.9, 150-199 14.3, 200-249 21.5, 250-349 28-32 (fit 1.0 + 0.097 ms per id). Init at boot 8 ms.

**Text -> first PCM chunk ready** (`textp`, front end + queue + engine, no I2S), 36 texts of 5-247 words, 154 utterances:

| | full split, int4 set | full split, int8 set | row split (either set) |
|---|---|---|---|
| median / p95 / max (ms) | 188.8 / 196.7 / 199.2 | 193.2 / 201.1 / 203.3 | 240-243 / 253 / 258 |
| texts under 200 ms | 36 of 36 | 32 of 36 | 0 of 16 |
| RTF (median per utterance; whole run) | 0.654; 0.656 | 0.667; 0.668 | 0.84-0.86 |
| underruns (emulated playout) | 0 | 0 | 0 |

Stand-alone run (240 other texts, 748 utterances, by length of the first utterance): 0-50 ids median 182 ms (max 188), 51-100 ids 189 (191), 101-150 ids 191 (196): **all 208 under 200 ms**; 151-250 ids 197 (up to 231), 251-400 ids 208 (up to 217).
500 short texts (2-9 words): median 183.2, p95 189.9, max 194.0 ms, 490 of 490 under 200 ms. So for typical sentences (up to about 20 words) the first chunk is ready in under 200 ms **from the end of the text** on the full split with the int4 set
(int8 set: 190-200 ms, borderline); on the row split it is 240-250 ms; for a long first sentence the front end adds 15-35 ms (up to 100 ms for a first piece over 400 ids, which say.py's rule phonemizes whole and then in halves).
The front end of sentence 2 and later runs in the background under the synthesis and does not change its RTF (0.655 for utterances 2+ vs 0.652 for the first; the next utterance was ready at least 266 ms before the previous one finished playing, 0 underruns in 118 transitions).
Policy: with `ITOFS_FE_FAST_FIRST` only the first sentence is phonemized before the engine starts, the following ones by the front-end task while the engine works; `fastfirst 0` gives exactly say.py's utterances (527/527 identical on the chip). Not implemented: splitting a very long first sentence at its first comma (it would change the ids from the host's).
No faster front-end path was needed for short sentences; none was found that keeps the ids identical to the host for long first sentences (the front end already costs only 0.1 ms per id).

## Stand-alone run (no computer) and robustness

`CONFIG_ITOFS_AUTOTEST` (with `_QUIET` and optionally `_LOOP_BOOTS=N`, `_LOOP_TEXTS=M`, `_SELFRESET`, `_WDT_RESETS`): texts are read from the data partition `texts` (flash 0xF80000, `tools/make_texts_bin.py out.bin file.tsv`), the chip phonemizes
and synthesizes them (no I2S, emulated playout clock) and logs one 64-byte record per utterance in `results` (0xFA0000; `tools/read_results.py`; `tools/host_compare.py` compares with the Python front end and `esp32/host/ito_batch`). Procedure: flash app + `texts.bin`, erase `results`,
`esptool --after watchdog_reset read-mac` to start, do not open the port, read `results` back later.
Measured: 240 texts (1-150 words, 20 multi-sentence paragraphs; none from the tuning corpus), 748 utterances, 5448 s of audio, no host attached while it ran (the log was read after a reset that interrupted it at 597 utterances; the firmware resumed): **ids 748/748 and PCM 748/748
identical to the host, 0 underruns, RTF 0.654**. Reboot loop: 20 boots (10 software resets, 9 hardware-watchdog resets deliberately provoked, 1 start), 60 texts, 180/180 identical; the first watchdog reset latched the row split as designed (results unchanged, RTF 0.83).
A true unplug could not be done; this proves the chip needs no computer to turn text into the exact audio the host would produce, not battery operation, a zero-volt cold start or the DAC path.
Fuzz: 1200 hostile inputs (random bytes, invalid UTF-8, lines up to 20 kB, Unicode, control codes, command look-alikes): no crash, hang or reset; a command sent while the front end is busy is refused (`ERROR front end busy (cancel first)`), not queued.

## Limits and what fails

- English (en-us) only. Foreign words get espeak's English guess; scripts that make espeak switch voice are not switched (matched the host on all tests, but not guaranteed).
- One utterance <= 2400 bytes after normalization and <= 400 tokens for the engine; the stream splitter handles long text at sentence ends and mid-word otherwise
  (say.py's rule); a *single* sentence over ~600 characters without any `.!?;` can give a first half over 400 tokens, which the engine cannot take (truncated).
- The tokenizer is a hand-written subset of nltk: exact on every test, but characters/patterns not in the tests could differ (e.g. nltk contraction rules for all-ASCII words).
- Matches *this* phonemizer/nltk/espeak-ng trio; the host result changes with those versions (see above).
- The `text` command takes one line (<= 4 KB) of UTF-8 over the console. While a text is being processed the console refuses the next one (`ERROR front end busy (cancel first)`); it is not queued.
- A single word of more than ~2400 bytes cannot be split and gives `ERROR front end: -2` (no speech, no crash).
- Latency over 200 ms from the end of the text: first sentences over about 150 ids (~20 words), the int8 set in some cases, and the whole row split (see "Measured on silicon").

## Reproduce

```
cd esp32/host && make test_frontend                   # C front end vs golden ids from Python (no weights, no Python)
cd esp32/frontend && make && python3 tools/make_reference.py corpus/corpus.tsv ref.jsonl && build/fe_cli data/espeak_en_us.bin corpus/corpus.tsv > c.jsonl \
  && python3 tools/diff_corpus.py ref.jsonl c.jsonl      # needs phonemizer, espeakng-loader, nltk (+punkt_tab)
tools/qemu_run.sh <fw build dir> models/ito_female_esp32s3.bin ref.jsonl --stream --fastfirst0   # real firmware in QEMU
tools/serial_ids.py --port /dev/cu.usbserial-XXXX ref.jsonl     # same on a board (`ids <text>`), also prints the chip's front-end ms (--no-wait if already booted)
# on silicon (this file's "Measured on silicon"): stream reference with the chip's default splitting, end-to-end timing, host PCM comparison, fuzz
python3 tools/make_stream_reference.py corpus/e2e.tsv ref_e2e.jsonl --fastfirst
python3 tools/serial_e2e.py --port PORT corpus/e2e.tsv e2e.jsonl --cmd textp [--tier 0]
make -C ../host ito_batch && python3 tools/host_compare.py --ref ref_e2e.jsonl --chip e2e.jsonl --weights0 ../../models/ito_female_esp32s3.bin --weights1 ../../models/ito_female_esp32s3_int4.bin
python3 tools/serial_fuzz.py --port PORT --n 1200 --seed 1
# stand-alone run: build with CONFIG_ITOFS_AUTOTEST=y (+ CONFIG_ITOFS_AUTOTEST_QUIET=y), then
python3 tools/make_texts_bin.py texts.bin corpus/autotest_texts.tsv corpus/autotest_paras.tsv
esptool write-flash 0xF80000 texts.bin ;  esptool erase-region 0xFA0000 0x40000 ;  (reset, wait for the run: about as long as the audio, ~90 min for the 240 texts)
esptool read-flash 0xFA0000 0x40000 results.bin && python3 tools/read_results.py results.bin results.jsonl
python3 tools/make_stream_reference.py corpus/autotest_texts.tsv+paras ref_at.jsonl --fastfirst && python3 tools/host_compare.py --ref ref_at.jsonl --chip results.jsonl --weights1 ...int4.bin
```
