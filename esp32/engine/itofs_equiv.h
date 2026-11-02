// Which weight sets are judged equivalent by ear, PER VOICE. The boot's set choice (itofs_sched_pick) looks at speed alone only among sets marked
// equivalent; a set that is not (or that this table does not know) is used only when no equivalent set meets the RTF target.
//
// A set is identified by the host engine's PCM hash of its self-test sentence (the fnv32 stored in the blob's "selftest" record, word 5): it differs
// for every voice and every set, so no global flag is involved. Validation status (the owner's blind listening tests):
//   male   : main int8, int4, light   -> equivalent
//   female : main int8, int4          -> equivalent;  light -> NOT validated by ear (an objective check was under way)
// A build or a stored override can add the unvalidated light sets (ITOFS_APPROVE_LIGHT), once the owner has approved them.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { ITOFS_KIND_MAIN = 0, ITOFS_KIND_INT4 = 1, ITOFS_KIND_LIGHT = 2 } itofs_set_kind_t;
#define ITOFS_APPROVE_LIGHT 1u         // override bit: treat the light sets of the table as equivalent

typedef struct {
    uint32_t id;                       // fnv32 of the host engine's PCM for the blob's self-test sentence (8-bit activations)
    const char *voice, *set;
    int kind;                          // itofs_set_kind_t
    int validated;                     // 1: judged equivalent by ear
} itofs_set_info_t;

// The table entry of a blob id, or NULL (an unknown blob: a retrained voice, a custom set).
const itofs_set_info_t *itofs_set_lookup(uint32_t id);
// 1 if the set may be chosen on speed alone: validated, or of an approved kind (approve = ITOFS_APPROVE_* bits). Unknown blobs: 0.
int itofs_set_equivalent(uint32_t id, unsigned approve);

#ifdef __cplusplus
}
#endif
