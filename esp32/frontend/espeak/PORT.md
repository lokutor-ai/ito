# Changes to espeak-ng 1.52.0 (https://github.com/espeak-ng/espeak-ng, GPLv3+) for the text->phonemes port
Source files copied unchanged unless listed. Edits are marked `ITO`.
- `common.c`: `GetFileLength` answers from the memory blob (`ito_vfs`).
- `synthdata.c`: `ReadPhFile` returns pointers into the blob (no copy); `FreePhData` no longer frees them.
- `dictionary.c`: `LoadDictionary` points `data_dictlist` into the blob; `translate.c` `DeleteTranslator` does not free it.
- `voices.c`: `LoadVoice` reads the voice text from the blob; voice listing/selection code is `#if 0`'d (the en-us voice `gmw/en-US` is loaded by name).
- added: `ito_vfs.[ch]`, `ito_stubs.c` (symbols of synthesize/wavegen/speech/soundicon/compiledict that translation references), `ito_espeak.[ch]` (init as espeak_ng_Initialize + SetVoiceByName; clause loop as espeak_TextToPhonemes), `config.h`, `setlengths.c` copied.
- `hostcompat`/`include/compat*`: the wctype shim maps iswalpha etc. to ucd-tools (as upstream does).
Not included: synthesis, wave generation, audio, mbrola, klatt, SSML voice switching, compile tools.
