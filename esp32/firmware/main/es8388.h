// ES8388 audio codec (I2C control, I2S audio) on the "ESP_S3-AUDIN" shield: DAC path to the speaker amplifier. Used only with CONFIG_ITOFS_AUDIO_ES8388.
#pragma once
#include <stdint.h>

// Probe the codec on the I2C bus (0x10 or 0x11) and configure it as an I2S slave playing 16-bit stereo at 256 x fs MCLK: reset, power, DAC on, output
// mixer on LOUT/ROUT, output volume (vol: 0..33, 30 = 0 dB, 1.5 dB per step), DAC muted. The I2S clocks (MCLK included) must already run.
// Returns 0 on success, -1 if no codec answers (nothing else is touched), -2 on a bus error.
int es8388_init(int sda, int scl, int vol);
void es8388_set_volume(int vol);         // output volume register of the four outputs, 0..33 (30 = 0 dB, 1.5 dB per step)
void es8388_mute(int mute);              // DAC digital mute, with the codec's own volume ramp
