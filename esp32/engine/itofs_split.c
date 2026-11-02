#include <stddef.h>
#include <string.h>
#include "itofs_split.h"

static uint32_t crc_of(const itofs_split_rec_t *r)
{
    const uint8_t *p = (const uint8_t *)r;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < offsetof(itofs_split_rec_t, crc); i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

void itofs_split_default(itofs_split_rec_t *r, int policy)
{
    memset(r, 0, sizeof *r);
    r->version = ITOFS_SPLIT_VERSION; r->policy = (uint8_t)policy; r->pref = (uint8_t)policy;
    r->verdict = ITOFS_VERDICT_UNTESTED; r->why = ITOFS_WHY_NONE;
}
int itofs_split_valid(const itofs_split_rec_t *r) { return r->version == ITOFS_SPLIT_VERSION && r->crc == crc_of(r); }
void itofs_split_seal(itofs_split_rec_t *r) { r->crc = crc_of(r); }

// the split the record asks for (before any test): forced full, or full for a passed auto verdict; a FAIL verdict always means the row split
int itofs_split_mode(const itofs_split_rec_t *r)
{
    if (r->pref == ITOFS_PREF_ROW || r->verdict == ITOFS_VERDICT_FAIL) return 0;
    return r->pref == ITOFS_PREF_FULL || r->verdict == ITOFS_VERDICT_PASS;
}

static void against(itofs_split_rec_t *r, itofs_why_t why)
{
    r->verdict = ITOFS_VERDICT_FAIL; r->why = (uint8_t)why; r->full_active = 0; r->testing = 0; r->fails++;
}

itofs_split_plan_t itofs_split_boot(itofs_split_rec_t *r, int policy, itofs_rst_t rst)
{
    itofs_split_plan_t p = { 0, 0, 0 };
    const itofs_split_rec_t before = *r;
    if (r->policy != policy) {                            // another build policy: restart preference and verdict, keep the counters
        r->policy = (uint8_t)policy; r->pref = (uint8_t)policy; r->verdict = ITOFS_VERDICT_UNTESTED; r->why = ITOFS_WHY_POLICY;
        r->testing = r->full_active = 0;
    }
    r->boots++;
    if (rst == ITOFS_RST_BROWNOUT) r->brownouts++;
    if (rst == ITOFS_RST_WDT || rst == ITOFS_RST_PANIC) r->wdt_resets++;
    if (rst != ITOFS_RST_BENIGN) {
        if (r->testing) against(r, rst == ITOFS_RST_BROWNOUT ? ITOFS_WHY_TEST_BROWNOUT : rst == ITOFS_RST_WDT ? ITOFS_WHY_TEST_HANG : ITOFS_WHY_TEST_PANIC);
        else if (r->full_active) against(r, rst == ITOFS_RST_BROWNOUT ? ITOFS_WHY_RUN_BROWNOUT : rst == ITOFS_RST_WDT ? ITOFS_WHY_RUN_WDT : ITOFS_WHY_RUN_PANIC);
    } else if (r->testing) {                              // power was cut or the board was reset during the test: no verdict, test again
        r->testing = 0; r->interrupted++;
    }
    const int want_full = itofs_split_mode(r);
    p.full = want_full;
    if (r->pref == ITOFS_PREF_AUTO && r->verdict != ITOFS_VERDICT_FAIL) {
        if (r->verdict == ITOFS_VERDICT_UNTESTED) { p.test = 2; p.full = 1; r->testing = 1; r->full_active = 0; }
        else p.test = 1;
    }
    if (p.test != 2) r->full_active = (uint8_t)want_full;
    p.save = memcmp(&before, r, offsetof(itofs_split_rec_t, crc)) != 0;
    return p;
}

int itofs_split_test_done(itofs_split_rec_t *r, int ok)
{
    r->testing = 0; r->tests++;
    if (ok) { r->passes++; r->verdict = ITOFS_VERDICT_PASS; r->why = ITOFS_WHY_TEST_PASS; r->full_active = 1; return 1; }
    against(r, ITOFS_WHY_TEST_MISMATCH); r->mismatches++;
    return 0;
}

void itofs_split_fail(itofs_split_rec_t *r, itofs_why_t why)
{
    against(r, why);
    if (why == ITOFS_WHY_RUN_MISMATCH || why == ITOFS_WHY_TEST_MISMATCH) r->mismatches++;
}

itofs_split_plan_t itofs_split_set_pref(itofs_split_rec_t *r, itofs_pref_t pref)
{
    itofs_split_plan_t p = { 0, 0, 1 };
    r->pref = (uint8_t)pref; r->testing = 0;
    r->verdict = ITOFS_VERDICT_UNTESTED;                  // the owner's command clears a latch
    if (pref == ITOFS_PREF_ROW) { r->why = ITOFS_WHY_USER_ROW; r->full_active = 0; }
    else if (pref == ITOFS_PREF_FULL) { r->why = ITOFS_WHY_USER_FULL; r->full_active = 1; p.full = 1; }
    else { r->why = ITOFS_WHY_USER_AUTO; r->full_active = 0; r->testing = 1; p.test = 2; p.full = 1; }
    return p;
}

const char *itofs_split_why_name(int w)
{
    static const char *const n[ITOFS_WHY_N] = { "none", "supply test passed", "supply test: hash mismatch", "supply test: hang (watchdog reset)", "supply test: brownout reset",
        "supply test: panic reset", "brownout reset in full split", "watchdog reset in full split", "panic reset in full split", "hash mismatch in full split",
        "owner set row", "owner set full", "owner set auto", "build policy changed" };
    return w >= 0 && w < ITOFS_WHY_N ? n[w] : "?";
}
const char *itofs_split_verdict_name(int v) { return v == ITOFS_VERDICT_PASS ? "pass" : v == ITOFS_VERDICT_FAIL ? "fail (row split latched)" : "untested"; }
const char *itofs_split_pref_name(int p) { return p == ITOFS_PREF_ROW ? "row" : p == ITOFS_PREF_FULL ? "full" : "auto"; }
