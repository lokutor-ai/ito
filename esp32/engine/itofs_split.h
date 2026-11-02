// Runtime choice between the two GEMM splits (row split: safe; full split: both cores in the int8 GEMM, faster, needs a stiff 3.3 V supply; see
// esp32/DUAL_CORE_POWER.md). Portable C99 with no hardware access: the firmware feeds it the reset reason and the results of its tests, stores the
// record in NVS, and the host test (host/host_test_split.c) drives it with simulated resets.
//
// Rules. The row split is the default until a supply self-test has passed. The full split runs only in "auto" (after a passed test, re-checked cheaply
// at every boot) or when forced by the owner (`split full`). ANY evidence against it while it is active (a brownout, watchdog or panic reset, a PCM
// hash mismatch) sets the verdict FAIL, which means the row split until the owner issues `split auto|row|full` again or erases the record.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITOFS_SPLIT_VERSION 1

typedef enum { ITOFS_RST_BENIGN = 0, ITOFS_RST_BROWNOUT, ITOFS_RST_WDT, ITOFS_RST_PANIC } itofs_rst_t;   // BENIGN: power-on, software, USB, external, deep sleep
typedef enum { ITOFS_PREF_AUTO = 0, ITOFS_PREF_ROW = 1, ITOFS_PREF_FULL = 2 } itofs_pref_t;
typedef enum { ITOFS_VERDICT_UNTESTED = 0, ITOFS_VERDICT_PASS = 1, ITOFS_VERDICT_FAIL = 2 } itofs_verdict_t;
typedef enum {
    ITOFS_WHY_NONE = 0, ITOFS_WHY_TEST_PASS, ITOFS_WHY_TEST_MISMATCH, ITOFS_WHY_TEST_HANG, ITOFS_WHY_TEST_BROWNOUT, ITOFS_WHY_TEST_PANIC,
    ITOFS_WHY_RUN_BROWNOUT, ITOFS_WHY_RUN_WDT, ITOFS_WHY_RUN_PANIC, ITOFS_WHY_RUN_MISMATCH,
    ITOFS_WHY_USER_ROW, ITOFS_WHY_USER_FULL, ITOFS_WHY_USER_AUTO, ITOFS_WHY_POLICY, ITOFS_WHY_N
} itofs_why_t;

// The persisted record (the firmware stores it as one NVS blob). crc covers every byte before it.
typedef struct {
    uint8_t version, policy;          // layout version; the build's initial policy when the record was made (a different build policy restarts the record)
    uint8_t pref, verdict, why;       // owner's preference, supply verdict, reason of the last change
    uint8_t testing;                  // 1 while the long supply test runs: a reset with this set means the test itself crashed the board
    uint8_t full_active;              // 1 while the full split is engaged: a bad reset with this set is held against it
    uint8_t reserved;
    uint32_t boots, tests, passes, fails, brownouts, wdt_resets, mismatches, interrupted;
    uint32_t crc;
} itofs_split_rec_t;

typedef struct {
    int test;                         // 0 none, 1 quick re-check of a passed verdict, 2 the long supply test (the record already has testing = 1)
    int full;                         // GEMM split to run: 1 full, 0 row (when test != 0: the split to run if the test passes)
    int save;                         // 1: the record changed and must be written before going on
} itofs_split_plan_t;

void itofs_split_default(itofs_split_rec_t *r, int policy);
int itofs_split_valid(const itofs_split_rec_t *r);                                       // version and crc
void itofs_split_seal(itofs_split_rec_t *r);                                             // recompute crc (call before storing)
// A boot: apply the evidence of the reset that led here (rst), restart the record if the build policy changed, say what to run.
itofs_split_plan_t itofs_split_boot(itofs_split_rec_t *r, int policy, itofs_rst_t rst);
// The long test finished (ok = bit-exact and no hang). Returns the split to run (1 full, 0 row).
int itofs_split_test_done(itofs_split_rec_t *r, int ok);
// Evidence against the full split during this run (hash mismatch, failed quick check). Latches the row split.
void itofs_split_fail(itofs_split_rec_t *r, itofs_why_t why);
// Owner command. Returns the plan to apply now (auto: the long test; full: forced, no test; row: row).
itofs_split_plan_t itofs_split_set_pref(itofs_split_rec_t *r, itofs_pref_t pref);
int itofs_split_mode(const itofs_split_rec_t *r);                                        // the split the record asks for now: 1 full, 0 row
const char *itofs_split_why_name(int why);
const char *itofs_split_verdict_name(int v);
const char *itofs_split_pref_name(int p);

#ifdef __cplusplus
}
#endif
