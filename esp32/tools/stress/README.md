# Two-core PIE stress test (ESP32-S3)

A small, self-contained ESP-IDF app (v5.5.1) that was run on the physical board on 9 October 2026. It is not part of the TTS firmware and contains no model weights.

What it does: it runs Ito's own int8 PIE dot kernel (`main/dot_rows_s3.S`, an unchanged copy of `esp32/firmware/main/dot_rows_s3.S`) on both cores at once. The activation tile is in internal SRAM and is read by both cores; random int8 weights are streamed from PSRAM through the data cache; the output channels are split in half between the cores. The int32 result of every call is hashed and compared with a one-core reference of the same data (the reference is computed twice, and a plain C check covers part of the first block). Core 1 is started by spinning on a flag; the real engine starts it with a FreeRTOS notification instead (a looser handshake). A start with both cores aligned to the same cycle was not tested.

Shapes (rows x K, outputs): K=704 (64 rows, 176 outputs split 88/88, 45 KB tile; the pattern the Oído project reported a fault with), K=576 (64 rows), K=576 and K=192 and K=1216 with the tile sizes of Ito's own layers, and a one-core control at K=704.

Build and flash (this touches a board, so only do it on one you own): `idf.py set-target esp32s3 && idf.py build flash monitor` in this directory. The sdkconfig defaults set 240 MHz, 8 MB octal PSRAM at 80 MHz and the 64 KB data cache, like the TTS firmware.

Result of the run of 9 October 2026 (one board, three full rounds of about 7 minutes, 21 minutes in all): 0 mismatches and 0 crashes in every configuration. The raw log is `esp32/logs/synthetic_stress_240MHz_2026-10-09.log`. This is one board, one run, one chip revision; it does not show that other boards behave the same.
