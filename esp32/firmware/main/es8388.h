// ES8388 audio codec (I2C control, I2S audio) on the "ESP_S3-AUDIN" shield: DAC path to the speaker amplifier. Used only with CONFIG_ITOFS_AUDIO_ES8388.
#pragma once
#include <stdint.h>

// Probe the codec on the I2C bus (0x10 or 0x11) and configure it as an I2S slave playing 16-bit stereo at 256 x fs MCLK: reset, power, DAC on, output
// mixer on LOUT/ROUT, output volume (vol_percent 0..100, see es8388_set_volume_percent), DAC muted. The I2S clocks (MCLK included) must already run.
// Returns 0 on success, -1 if no codec answers (nothing else is touched), -2 on a bus error.
int es8388_init(int sda, int scl, int vol_percent);
void es8388_set_volume(int vol);         // output volume register of the four outputs, 0..33 (30 = 0 dB, 1.5 dB per step)
void es8388_set_volume_percent(int pct); // 0..100 -> register 0..33 (91 % = 0 dB, 100 % = +4.5 dB); 0 is a true mute that stays until a volume above 0 is set
void es8388_mute(int mute);              // DAC digital mute, with the codec's own volume ramp (about 60 ms from full scale to silence)
#ifdef CONFIG_ITOFS_AUDIO_LOOPBACK
// Test path: the shield's microphones go to the codec's ADC. Power the ADC on, I2S slave, 16 bit, 256 x fs; input `sel` (0: LIN1/RIN1, 1: LIN2/RIN2, 2: differential LIN1-RIN1, 3: differential LIN2-RIN2), PGA gain `gain` (0..8 = 0..24 dB).
int es8388_adc_init(int sel, int gain);
#endif
