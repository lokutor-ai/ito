#include <stdio.h>
#include <string.h>
#include "esp_system.h"
#include "nvs_flash.h"
#include "nvs.h"
#ifndef ITOFS_QEMU
#include "hal/wdt_hal.h"
#include "soc/rtc.h"
#endif
#include "guard.h"

#define NVS_NS "ito"
#define NVS_KEY "split"

itofs_rst_t guard_reset_class(const char **name)
{
    const esp_reset_reason_t r = esp_reset_reason();
    static const char *const n[] = { "unknown", "power-on", "external pin", "software", "panic", "interrupt watchdog", "task watchdog", "other watchdog",
                                     "deep sleep", "brownout", "SDIO", "USB", "JTAG", "eFuse", "power glitch", "CPU lockup" };
    if (name) *name = (unsigned)r < sizeof n / sizeof n[0] ? n[r] : "?";
    switch (r) {
    case ESP_RST_BROWNOUT: return ITOFS_RST_BROWNOUT;
    case ESP_RST_WDT: case ESP_RST_INT_WDT: case ESP_RST_TASK_WDT: return ITOFS_RST_WDT;
    case ESP_RST_PANIC: return ITOFS_RST_PANIC;
    default: return ITOFS_RST_BENIGN;
    }
}

static int nvs_begin(nvs_handle_t *h, nvs_open_mode_t mode)
{
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) { nvs_flash_erase(); e = nvs_flash_init(); }
    if (e != ESP_OK) return -1;
    e = nvs_open(NVS_NS, mode, h);
    if (e != ESP_OK) { nvs_flash_deinit(); return -1; }
    return 0;
}

int guard_rec_load(itofs_split_rec_t *r)
{
    nvs_handle_t h;
    if (nvs_begin(&h, NVS_READWRITE)) return -1;      // read-write: a blank partition is formatted by the first init
    itofs_split_rec_t t;
    size_t n = sizeof t;
    const int ok = nvs_get_blob(h, NVS_KEY, &t, &n) == ESP_OK && n == sizeof t && itofs_split_valid(&t);
    nvs_close(h); nvs_flash_deinit();
    if (ok) *r = t;
    return ok ? 0 : -1;
}

int guard_rec_save(itofs_split_rec_t *r)
{
    itofs_split_seal(r);
    nvs_handle_t h;
    if (nvs_begin(&h, NVS_READWRITE)) return -1;
    const int ok = nvs_set_blob(h, NVS_KEY, r, sizeof *r) == ESP_OK && nvs_commit(h) == ESP_OK;
    nvs_close(h); nvs_flash_deinit();
    return ok ? 0 : -1;
}

#ifndef ITOFS_QEMU
static wdt_hal_context_t s_rwdt = RWDT_HAL_CONTEXT_DEFAULT();
void guard_wdt_arm(uint32_t ms)
{
    const uint64_t hz = rtc_clk_slow_freq_get_hz();
    wdt_hal_init(&s_rwdt, WDT_RWDT, 0, false);                 // as ESP-IDF's panic handler arms it: reset enables set, interrupts off
    wdt_hal_write_protect_disable(&s_rwdt);
    // stage 0: reset the CPUs (reset reason "watchdog"; the system reset action did not fire on the board); stage 1, if the board is still not fed 5 s later: reset the chip
    wdt_hal_config_stage(&s_rwdt, WDT_STAGE0, (uint32_t)(ms * hz / 1000u), WDT_STAGE_ACTION_RESET_CPU);
    wdt_hal_config_stage(&s_rwdt, WDT_STAGE1, (uint32_t)(5000u * hz / 1000u), WDT_STAGE_ACTION_RESET_RTC);
    wdt_hal_enable(&s_rwdt);
    wdt_hal_write_protect_enable(&s_rwdt);
}
void guard_wdt_feed(void)
{
    wdt_hal_write_protect_disable(&s_rwdt);
    wdt_hal_feed(&s_rwdt);
    wdt_hal_write_protect_enable(&s_rwdt);
}
#else
void guard_wdt_arm(uint32_t ms) { (void)ms; }
void guard_wdt_feed(void) {}
#endif
