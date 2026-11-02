#include <stddef.h>
#include "itofs_equiv.h"

static const itofs_set_info_t s_sets[] = {
    { 0x023bd6b1u, "female", "main int8", ITOFS_KIND_MAIN,  1 },
    { 0x363d7d98u, "female", "int4",      ITOFS_KIND_INT4,  1 },
    { 0x5f12d5e1u, "female", "light",     ITOFS_KIND_LIGHT, 0 },       // not validated by ear
    { 0x35f96c0bu, "male",   "main int8", ITOFS_KIND_MAIN,  1 },
    { 0xb46d58deu, "male",   "int4",      ITOFS_KIND_INT4,  1 },
    { 0xa14d1c89u, "male",   "light",     ITOFS_KIND_LIGHT, 1 },
};

const itofs_set_info_t *itofs_set_lookup(uint32_t id)
{
    for (size_t i = 0; i < sizeof s_sets / sizeof s_sets[0]; i++) if (s_sets[i].id == id) return &s_sets[i];
    return NULL;
}

int itofs_set_equivalent(uint32_t id, unsigned approve)
{
    const itofs_set_info_t *s = itofs_set_lookup(id);
    if (!s) return 0;
    return s->validated || (s->kind == ITOFS_KIND_LIGHT && (approve & ITOFS_APPROVE_LIGHT));
}
