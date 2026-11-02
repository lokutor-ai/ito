#include <stdio.h>
#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "es8388.h"

enum { R_CONTROL1 = 0x00, R_CONTROL2 = 0x01, R_CHIPPOWER = 0x02, R_ADCPOWER = 0x03, R_DACPOWER = 0x04, R_MASTERMODE = 0x08, R_DACCONTROL1 = 0x17, R_DACCONTROL2 = 0x18,
       R_DACCONTROL3 = 0x19, R_DACCONTROL4 = 0x1A, R_DACCONTROL5 = 0x1B, R_DACCONTROL16 = 0x26, R_DACCONTROL17 = 0x27, R_DACCONTROL20 = 0x2A, R_DACCONTROL21 = 0x2B,
       R_DACCONTROL23 = 0x2D, R_LOUT1VOL = 0x2E, R_ROUT1VOL = 0x2F, R_LOUT2VOL = 0x30, R_ROUT2VOL = 0x31 };

static i2c_master_dev_handle_t s_dev;
static volatile int s_zero;                      // volume 0 is a true mute: the DAC stays muted, es8388_mute(0) of an utterance does not lift it

static int wr(uint8_t reg, uint8_t val)
{
    if (!s_dev) return -2;                       // no codec answered at start-up
    const uint8_t b[2] = { reg, val };
    return i2c_master_transmit(s_dev, b, 2, 100) == ESP_OK ? 0 : -2;
}

int es8388_init(int sda, int scl, int vol_percent)
{
    i2c_master_bus_handle_t bus;
    const i2c_master_bus_config_t bc = { .i2c_port = I2C_NUM_0, .sda_io_num = sda, .scl_io_num = scl, .clk_source = I2C_CLK_SRC_DEFAULT,
                                         .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true };
    if (i2c_new_master_bus(&bc, &bus) != ESP_OK) return -2;
    int addr = 0;
    for (int a = 0x10; a <= 0x11 && !addr; a++) if (i2c_master_probe(bus, (uint16_t)a, 50) == ESP_OK) addr = a;
    if (!addr) { i2c_del_master_bus(bus); return -1; }
    const i2c_device_config_t dc = { .dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = (uint16_t)addr, .scl_speed_hz = 100000 };
    if (i2c_master_bus_add_device(bus, &dc, &s_dev) != ESP_OK) return -2;
    printf("ES8388 codec answers at I2C 0x%02x (SDA %d, SCL %d)\n", addr, sda, scl);
    int e = 0;
    e |= wr(R_DACCONTROL3, 0x24);                // DAC muted (soft ramp on) while it is configured and until the first utterance
    e |= wr(R_CONTROL2, 0x50);
    e |= wr(R_CHIPPOWER, 0x00);                  // power up every block
    e |= wr(R_MASTERMODE, 0x00);                 // I2S slave: the ESP32 makes BCLK / LRCK / MCLK
    e |= wr(R_DACPOWER, 0xC0);                   // DAC and outputs off while the mixer is set
    e |= wr(R_CONTROL1, 0x12);
    e |= wr(R_DACCONTROL1, 0x18);                // I2S format, 16 bit
    e |= wr(R_DACCONTROL2, 0x02);                // MCLK = 256 x fs
    e |= wr(R_DACCONTROL4, 0x00); e |= wr(R_DACCONTROL5, 0x00);      // digital volume 0 dB
    e |= wr(R_DACCONTROL16, 0x00);
    e |= wr(R_DACCONTROL17, 0x90); e |= wr(R_DACCONTROL20, 0x90);    // left DAC to the left mixer, right DAC to the right mixer
    e |= wr(R_DACCONTROL21, 0x80);               // the DAC uses the ADC's LRCK
    e |= wr(R_DACCONTROL23, 0x00);
    es8388_set_volume_percent(vol_percent);
    e |= wr(R_CHIPPOWER, 0xF0); e |= wr(R_CHIPPOWER, 0x00);          // restart the state machine
    e |= wr(R_ADCPOWER, 0xFF);                   // the microphone path stays off (not used)
    e |= wr(R_DACPOWER, 0x3C);                   // DAC and LOUT1 / ROUT1 / LOUT2 / ROUT2 on
    return e ? -2 : 0;
}

void es8388_set_volume(int vol)
{
    if (vol < 0) vol = 0;
    if (vol > 33) vol = 33;
    wr(R_LOUT1VOL, (uint8_t)vol); wr(R_ROUT1VOL, (uint8_t)vol); wr(R_LOUT2VOL, (uint8_t)vol); wr(R_ROUT2VOL, (uint8_t)vol);
}

void es8388_set_volume_percent(int pct)
{
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    s_zero = pct == 0;
    es8388_set_volume((pct * 33 + 50) / 100);
    if (s_zero) wr(R_DACCONTROL3, 0x24);          // the volume register alone only reaches -45 dB
}

void es8388_mute(int mute) { wr(R_DACCONTROL3, mute || s_zero ? 0x24 : 0x20); }   // bit 5: the codec ramps its digital volume instead of stepping it (no pop)

#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
int es8388_adc_init(int sel, int gain)
{
    if (gain < 0) gain = 0;
    if (gain > 8) gain = 8;
    int e = 0;
    e |= wr(0x03, 0xFF);                         // ADC blocks off while they are set
    e |= wr(0x09, (uint8_t)((gain << 4) | gain)); // PGA gain, left and right
    e |= wr(0x0A, sel == 1 ? 0x50 : sel >= 2 ? 0xF0 : 0x00);   // input pair: 0 LIN1/RIN1, 1 LIN2/RIN2, 2 and 3 differential
    e |= wr(0x0B, sel == 3 ? 0x82 : 0x02);       // 3: the differential pair LIN2-RIN2 instead of LIN1-RIN1
    e |= wr(0x0C, 0x0C);                         // I2S, 16 bit
    e |= wr(0x0D, 0x02);                         // MCLK = 256 x fs
    e |= wr(0x10, 0x00); e |= wr(0x11, 0x00);    // digital volume 0 dB
    e |= wr(0x03, 0x00);                         // ADC on
    return e ? -2 : 0;
}
#endif
