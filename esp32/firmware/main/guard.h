// Runtime guards of the firmware: the reset reason, the hardware watchdog that turns a hang into a reset, and the NVS record of the split verdict.
#pragma once
#include <stdint.h>
#include "itofs_split.h"

itofs_rst_t guard_reset_class(const char **name);   // class of the reset that led to this boot (and its ESP-IDF name)
int guard_rec_load(itofs_split_rec_t *r);           // 0 and *r filled if a valid record is stored, else -1
int guard_rec_save(itofs_split_rec_t *r);           // seals and stores the record (NVS is opened and closed around the write); 0 on success
// The RTC watchdog resets the whole chip (reset reason "watchdog") when it is not fed for ms: it works when interrupts are off or a core is stuck.
// The synthesis loops feed it once per chunk and the idle loop once a second. No effect in the QEMU build (the emulator has no such hang).
void guard_wdt_arm(uint32_t ms);
void guard_wdt_feed(void);
