/* Ito on-chip front end: English text -> Ito token ids, the same ids as the host front end (ito/text.py).
 *
 *   text -> normalize (ito.text.normalize) -> strip, remove '"'
 *        -> phonemizer 3.x espeak backend logic (punctuation preserved, stress kept, en-us), with espeak-ng 1.52.0 itself
 *        -> nltk word_tokenize on the phoneme string -> StyleTTS 2 symbol table, leading 0
 *
 * Portable C99 (host and ESP-IDF). The espeak-ng data comes from one memory blob (tools/pack_data.py).
 * Copyright (C) 2026 Lokutor / Ito authors. GPLv3 or later, see ../../LICENSE; espeak-ng (src/espeak) is GPLv3 or later, see espeak/COPYING.
 * The symbol table is from StyleTTS 2 (MIT, (c) 2023 Aaron (Yinghao) Li). */
#ifndef ITO_FRONTEND_H
#define ITO_FRONTEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define ITOFE_MAX_TOKENS 400        /* the engine's limit per utterance (leading 0 included in the count the host uses) */
#define ITOFE_GROUP_TOKENS 200      /* consecutive short sentences are merged up to this many tokens (tools/say.py) */
#define ITOFE_MAX_TEXT 2400         /* bytes of one utterance after normalisation; longer text is split by the stream API */

#define ITOFE_OK 0
#define ITOFE_E_BLOB (-1)
#define ITOFE_E_TOO_LONG (-2)
#define ITOFE_E_ESPEAK (-3)
#define ITOFE_E_NOMEM (-4)

/* Memory hook for the working buffers (a few tens of KB). On the chip they go to PSRAM; default malloc. */
#ifndef ITOFE_ALLOC
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#define ITOFE_ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#else
#define ITOFE_ALLOC(n) malloc(n)
#endif
#endif

/* Initialise espeak-ng from the data blob and allocate working memory. Returns ITOFE_OK. */
int itofe_init(const void *blob, size_t len);

/* Whole pipeline for ONE utterance. ids[0] is the 0 pad. Returns the number of ids (>= 1) or a negative error. */
int itofe_text_to_ids(const char *text, uint8_t *ids, int cap);

/* Stages, for tests: */
int itofe_normalize(const char *text, char *out, size_t cap);                 /* ito.text.normalize + strip + remove '"' */
int itofe_phonemizer(const char *norm, char *out, size_t cap);                /* phonemizer output (before word_tokenize) */
int itofe_tokenize(const char *raw, char *out, size_t cap);                   /* " ".join(word_tokenize(raw)) */
int itofe_symbols(const char *tok, uint8_t *ids, int cap);                    /* leading 0 + symbol ids */

/* Streaming: split a long text into utterances exactly like tools/say.py (sentences at [.!?;] + whitespace, middle-word split above
 * 400 tokens, merge up to 200 tokens). One call = one utterance, its phonemization is done in that call. */
typedef struct {
    const char *text;       /* NUL-terminated, must stay valid while streaming */
    size_t pos;             /* next byte of the text to read */
    int fast_first;         /* 1: the first utterance is the first sentence alone (no look-ahead merge): lowest first-audio latency */
    int n_out;              /* utterances returned so far */
    char cur[ITOFE_MAX_TEXT + 8];            /* remainder of a sentence that had to be split (more than 400 tokens) */
    int have_next;          /* a look-ahead piece is stored */
    int next_n;
    uint8_t next_ids[ITOFE_MAX_TOKENS + 16];
    char next_text[ITOFE_MAX_TEXT + 8];
    char gtext[ITOFE_MAX_TEXT + 8];          /* text of the utterance being assembled */
} itofe_stream;

void itofe_stream_begin(itofe_stream *s, const char *text, int fast_first);
/* Returns the number of ids of the next utterance (> 0), 0 when the text is finished, negative on error.
 * fast_first = 0 gives exactly the utterances tools/say.py sends; fast_first = 1 only differs in that the first sentence is not merged
 * with the following ones (the rest follows say.py's grouping). */
int itofe_stream_next(itofe_stream *s, uint8_t *ids, int cap);

/* Used by the tests / memory report */
size_t itofe_work_bytes(void);

#ifdef __cplusplus
}
#endif
#endif
