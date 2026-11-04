#!/bin/bash
# Boot the QEMU build of the firmware (build dir $1, default /tmp/fwb_qemu) with a weights blob and run serial_ids.py against it.
#   qemu_run.sh <fwbuild> <weights.bin> <ref.jsonl> [serial_ids.py args...]
set -e
FB=${1:-/tmp/fwb_qemu}; W=${2:-$(dirname "$0")/../../../models/ito_female_esp32s3.bin}; REF=$3; shift 3
IDF=${IDF_PATH:-$HOME/esp/esp-idf}; . "$IDF/export.sh" > /dev/null
Q=$(ls $HOME/.espressif/tools/qemu-xtensa/*/qemu/bin/qemu-system-xtensa | head -1)
mkdir -p $FB/run
python3 - <<PY
import sys; sys.path.insert(0, "$IDF/tools")
from idf_py_actions.qemu_ext import QEMU_TARGETS
open("$FB/run/qemu_efuse.bin", "wb").write(QEMU_TARGETS["esp32s3"].default_efuse)
PY
python -m esptool --chip esp32s3 merge_bin --fill-flash-size 16MB -o $FB/run/flash.bin --flash_mode dio --flash_size 16MB \
  0x0 $FB/bootloader/bootloader.bin 0x8000 $FB/partition_table/partition-table.bin 0x10000 $FB/itofs.bin 0x200000 "$W" > /dev/null
PORT=$((5500 + RANDOM % 400))
"$Q" -M esp32s3 -m 8M $QEMU_EXTRA -nographic -monitor none -nic none \
  -drive file=$FB/run/flash.bin,if=mtd,format=raw \
  -drive file=$FB/run/qemu_efuse.bin,if=none,format=raw,id=efuse -global driver=nvram.esp32s3.efuse,property=drive,value=efuse \
  -global driver=timer.esp32s3.timg,property=wdt_disable,value=true \
  -serial tcp:127.0.0.1:$PORT,server=on,wait=on > $FB/run/qemu.stdout 2>&1 &
QPID=$!
trap "kill $QPID 2>/dev/null || true" EXIT
sleep 1
python3 "$(dirname "$0")/serial_ids.py" --tcp 127.0.0.1:$PORT "$REF" "$@"
