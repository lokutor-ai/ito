/* ITO: text -> phonemes with espeak-ng 1.52.0 (en-us), as phonemizer's EspeakWrapper.text_to_phonemes() calls it. GPLv3. */
#ifndef ITO_ESPEAK_H
#define ITO_ESPEAK_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* blob: see ito_vfs.h. Returns 0 on success. Loads the en-us voice ("gmw/en-US"). */
int itoes_init(const void *blob, size_t len);
/* One chunk of text (no sentence punctuation expected, but any text works) -> IPA with '_' between phonemes, clauses joined by ' '.
 * Same as phonemizer: espeak_TextToPhonemes(&p, UTF8, '_'<<8 | 2) repeated until the text is used up. Returns the length or -1. */
int itoes_text_to_phonemes(const char *utf8, char *out, size_t cap);
#ifdef __cplusplus
}
#endif
#endif
