// Host test of engine/itofs_split.c: the verdict state machine with simulated resets and test results (no board needed).
//   make test_split
#include <stdio.h>
#include <string.h>
#include "itofs_split.h"

static int fails;
#define CHECK(c, what) do { if (!(c)) { printf("FAIL: %s (line %d)\n", what, __LINE__); fails++; } } while (0)

// one simulated boot of a board with a record stored in "NVS": returns the split that ends up running; test_ok: result of a long/quick test if one is asked for
static itofs_split_rec_t nvs;
static int boot(int policy, itofs_rst_t rst, int test_ok)
{
    itofs_split_rec_t r = nvs;
    if (!itofs_split_valid(&r)) itofs_split_default(&r, policy);
    const itofs_split_plan_t p = itofs_split_boot(&r, policy, rst);
    int full = p.full;
    if (p.save) { itofs_split_seal(&r); nvs = r; }
    if (p.test == 2) { full = itofs_split_test_done(&r, test_ok); itofs_split_seal(&r); nvs = r; }
    else if (p.test == 1 && !test_ok) { itofs_split_fail(&r, ITOFS_WHY_RUN_MISMATCH); full = 0; itofs_split_seal(&r); nvs = r; }
    return full;
}
static void erase(void) { memset(&nvs, 0, sizeof nvs); }
static itofs_split_rec_t get(void) { return nvs; }
static void user(itofs_pref_t pref, int test_ok, int *full)
{
    itofs_split_rec_t r = nvs;
    itofs_split_plan_t p = itofs_split_set_pref(&r, pref);
    *full = p.test == 2 ? itofs_split_test_done(&r, test_ok) : p.full;
    itofs_split_seal(&r); nvs = r;
}

int main(void)
{
    int full;
    // 1. first boot in auto: the long test decides; a pass engages the full split, later boots re-check cheaply
    erase();
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1, "first boot, test passes -> full");
    CHECK(get().verdict == ITOFS_VERDICT_PASS && get().why == ITOFS_WHY_TEST_PASS && get().full_active && !get().testing, "pass recorded");
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1, "second boot: quick check passes -> full");
    CHECK(get().tests == 1 && get().boots == 2, "counters");
    // 2. a failed test: row, latched, no new test on later boots
    erase();
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 0) == 0, "test fails -> row");
    CHECK(get().verdict == ITOFS_VERDICT_FAIL && get().why == ITOFS_WHY_TEST_MISMATCH && get().mismatches == 1, "mismatch recorded");
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 0, "stays row after a failed test (latched)");
    // 3. the long test hangs: the board resets by watchdog with testing = 1 -> FAIL, row
    erase();
    { itofs_split_rec_t r; itofs_split_default(&r, ITOFS_PREF_AUTO); itofs_split_plan_t p = itofs_split_boot(&r, ITOFS_PREF_AUTO, ITOFS_RST_BENIGN);
      CHECK(p.test == 2 && r.testing, "long test armed"); itofs_split_seal(&r); nvs = r; }
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_WDT, 1) == 0, "reset by watchdog during the test -> row");
    CHECK(get().verdict == ITOFS_VERDICT_FAIL && get().why == ITOFS_WHY_TEST_HANG && get().wdt_resets == 1 && !get().testing, "hang recorded");
    // 3b. same with brownout and panic
    erase(); { itofs_split_rec_t r; itofs_split_default(&r, 0); itofs_split_boot(&r, 0, ITOFS_RST_BENIGN); itofs_split_seal(&r); nvs = r; }
    boot(ITOFS_PREF_AUTO, ITOFS_RST_BROWNOUT, 1);
    CHECK(get().why == ITOFS_WHY_TEST_BROWNOUT && get().brownouts == 1, "brownout during the test");
    erase(); { itofs_split_rec_t r; itofs_split_default(&r, 0); itofs_split_boot(&r, 0, ITOFS_RST_BENIGN); itofs_split_seal(&r); nvs = r; }
    boot(ITOFS_PREF_AUTO, ITOFS_RST_PANIC, 1);
    CHECK(get().why == ITOFS_WHY_TEST_PANIC, "panic during the test");
    // 3c. power cut during the test (benign reset): no verdict, tested again
    erase(); { itofs_split_rec_t r; itofs_split_default(&r, 0); itofs_split_boot(&r, 0, ITOFS_RST_BENIGN); itofs_split_seal(&r); nvs = r; }
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1, "interrupted test is repeated and passes");
    CHECK(get().interrupted == 1 && get().verdict == ITOFS_VERDICT_PASS, "interrupted counted");
    // 4. any bad reset while the full split runs latches the row split
    const itofs_rst_t bad[] = { ITOFS_RST_BROWNOUT, ITOFS_RST_WDT, ITOFS_RST_PANIC };
    const itofs_why_t why[] = { ITOFS_WHY_RUN_BROWNOUT, ITOFS_WHY_RUN_WDT, ITOFS_WHY_RUN_PANIC };
    for (int i = 0; i < 3; i++) {
        erase(); boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1);
        CHECK(boot(ITOFS_PREF_AUTO, bad[i], 1) == 0, "bad reset in full -> row");
        CHECK(get().verdict == ITOFS_VERDICT_FAIL && get().why == why[i] && !get().full_active, "bad reset recorded");
        CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 0 && boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 0, "stays row for good");
    }
    // 5. power-on, software and USB resets do not count against the full split
    erase(); boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1);
    for (int i = 0; i < 5; i++) CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1, "benign resets keep the full split");
    // 6. a hash mismatch (quick check or self-test) in full latches the row split
    erase(); boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1);
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 0) == 0, "quick check fails -> row");
    CHECK(get().why == ITOFS_WHY_RUN_MISMATCH && get().mismatches == 1, "run mismatch recorded");
    { itofs_split_rec_t r = nvs; itofs_split_fail(&r, ITOFS_WHY_RUN_MISMATCH); CHECK(itofs_split_mode(&r) == 0, "mode row after fail"); }
    // 7. owner commands clear the latch
    user(ITOFS_PREF_FULL, 1, &full);
    CHECK(full == 1 && get().verdict == ITOFS_VERDICT_UNTESTED && get().full_active, "split full forces full without a test");
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1, "forced full persists across a boot (pref is stored)");
    user(ITOFS_PREF_ROW, 1, &full);
    CHECK(full == 0 && boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 0 && get().tests >= 1, "split row persists");
    user(ITOFS_PREF_AUTO, 1, &full);
    CHECK(full == 1 && get().verdict == ITOFS_VERDICT_PASS, "split auto re-runs the test and passes");
    user(ITOFS_PREF_AUTO, 0, &full);
    CHECK(full == 0 && get().verdict == ITOFS_VERDICT_FAIL, "split auto with a failing test -> row");
    // 8. a forced full that is browned out falls back to row (no boot loop); the owner must ask again
    user(ITOFS_PREF_FULL, 1, &full);
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BROWNOUT, 1) == 0 && get().verdict == ITOFS_VERDICT_FAIL, "forced full + brownout -> row");
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 0, "no boot loop");
    // 9. a bad reset in the row split is only counted
    erase(); boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1); user(ITOFS_PREF_ROW, 1, &full);
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BROWNOUT, 1) == 0 && get().brownouts == 1 && get().verdict == ITOFS_VERDICT_UNTESTED, "brownout in row: counted, no verdict");
    // 10. a corrupt or old record restarts; a changed build policy restarts verdict and preference but keeps counters
    erase(); boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1);
    nvs.pref ^= 1; CHECK(!itofs_split_valid(&nvs), "crc catches a flipped byte");
    CHECK(boot(ITOFS_PREF_AUTO, ITOFS_RST_BENIGN, 1) == 1 && get().boots == 1, "corrupt record -> defaults, new test");
    { itofs_split_rec_t r = nvs; r.version = 0; itofs_split_seal(&r); CHECK(!itofs_split_valid(&r), "old version rejected"); }
    CHECK(boot(ITOFS_PREF_ROW, ITOFS_RST_BENIGN, 1) == 0 && get().why == ITOFS_WHY_POLICY && get().boots == 2, "build policy row: restart, counters kept");
    // 11. the build's initial policy: full forces without a test, row never tests
    erase(); CHECK(boot(ITOFS_PREF_FULL, ITOFS_RST_BENIGN, 0) == 1 && get().tests == 0, "policy full: no test");
    erase(); CHECK(boot(ITOFS_PREF_ROW, ITOFS_RST_BENIGN, 1) == 0 && get().tests == 0, "policy row: no test");
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
