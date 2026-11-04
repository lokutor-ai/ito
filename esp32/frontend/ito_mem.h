/* ITO: force-included into the espeak-ng sources of the ESP-IDF build so that every heap allocation of the translator goes to PSRAM
 * (internal SRAM is almost all used by the synthesis engine). The host build does not include it. */
#ifndef ITO_MEM_H
#define ITO_MEM_H
#include <stdlib.h>
#include <string.h>
void *itofe_malloc(size_t n);
void *itofe_calloc(size_t a, size_t b);
void *itofe_realloc(void *p, size_t n);
void itofe_free(void *p);
char *itofe_strdup(const char *s);
#define malloc itofe_malloc
#define calloc itofe_calloc
#define realloc itofe_realloc
#define free itofe_free
#define strdup itofe_strdup
#endif
