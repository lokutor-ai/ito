/* ESP-IDF glue of the on-chip front end: PSRAM allocation, the embedded espeak-ng data blob, init. GPLv3 or later. */
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "ito_frontend.h"
#include "ito_fe_idf.h"

#undef malloc
void *itofe_malloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
void *itofe_calloc(size_t a, size_t b) { return heap_caps_calloc(a, b, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
void *itofe_realloc(void *p, size_t n) { return heap_caps_realloc(p, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
void itofe_free(void *p) { heap_caps_free(p); }
char *itofe_strdup(const char *s) { size_t l = strlen(s) + 1; char *d = itofe_malloc(l); if (d) memcpy(d, s, l); return d; }

extern const uint8_t espeak_blob_start[] asm("_binary_espeak_en_us_bin_start");
extern const uint8_t espeak_blob_end[] asm("_binary_espeak_en_us_bin_end");

int itofe_idf_init(void)
{
    return itofe_init(espeak_blob_start, (size_t)(espeak_blob_end - espeak_blob_start));
}
size_t itofe_idf_blob_bytes(void) { return (size_t)(espeak_blob_end - espeak_blob_start); }
