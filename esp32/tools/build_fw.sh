#!/bin/bash
# Build the firmware with ESP-IDF 5.5 (IDF_PATH, default ~/esp/esp-idf): the hardware configuration (octal PSRAM, QIO
# flash, I2S on) and the QEMU configuration (quad PSRAM, DIO flash, I2S compiled out), then write
# esp32/prebuilt/ito_app_merged.bin (bootloader + partition table + app; flash it at 0x0).
#   esp32/tools/build_fw.sh
set -e
cd "$(dirname "$0")/../firmware"
IDF=${IDF_PATH:-$HOME/esp/esp-idf}
. "$IDF/export.sh" > /dev/null
idf.py -B build_hw -D SDKCONFIG=build_hw/sdkconfig set-target esp32s3 > /dev/null
idf.py -B build_hw -D SDKCONFIG=build_hw/sdkconfig build 2>&1 | tail -3
echo HW_BUILD_DONE
QD="sdkconfig.defaults;sdkconfig.qemu.defaults"
idf.py -B build_qemu -D ITOFS_QEMU=1 -D SDKCONFIG=build_qemu/sdkconfig -D SDKCONFIG_DEFAULTS="$QD" set-target esp32s3 > /dev/null
idf.py -B build_qemu -D ITOFS_QEMU=1 -D SDKCONFIG=build_qemu/sdkconfig -D SDKCONFIG_DEFAULTS="$QD" build 2>&1 | tail -3
echo QEMU_BUILD_DONE
python -m esptool --chip esp32s3 merge_bin -o ../prebuilt/ito_app_merged.bin --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x0 build_hw/bootloader/bootloader.bin 0x8000 build_hw/partition_table/partition-table.bin 0x10000 build_hw/itofs.bin
ls -la ../prebuilt/ito_app_merged.bin build_hw/itofs.bin build_qemu/itofs.bin
