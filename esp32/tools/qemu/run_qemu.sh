#!/bin/bash
# Run the QEMU build of the firmware (esp32/firmware/build_qemu, made by tools/build_fw.sh) with a weight blob in
# Espressif QEMU, drive it over a TCP serial port with qemu_client.py (boot self-test + demos, then the QEMU_SCRIPT
# commands), compare every PCM the firmware dumps with the host C engine's, then stop QEMU.
#
#   esp32/tools/qemu/run_qemu.sh [weights.bin] [tag]          (log in esp32/logs/qemu_<tag>.log)
#   VOICE=male esp32/tools/qemu/run_qemu.sh                male-voice blob from models/
#
# Needs ESP-IDF 5.5 (IDF_PATH, default ~/esp/esp-idf) and Espressif's QEMU >= 9.2.2 for esp32s3 (QEMU_BIN; older
# builds find no PSRAM). Timings printed under QEMU are emulator wall-clock times, NOT chip times.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
[ -n "$1" ] || "$ROOT/esp32/tools/fetch_weights.sh"
case "${VOICE:-female}" in g|G|m|M|male) WDEF=ito_male_esp32s3.bin ;; *) WDEF=ito_female_esp32s3.bin ;; esac
W=${1:-$ROOT/models/$WDEF}; TAG=${2:-v3}
FW=$ROOT/esp32/firmware; HOST=$ROOT/esp32/host; LOGS=$ROOT/esp32/logs
IDF=${IDF_PATH:-$HOME/esp/esp-idf}
HOST_PY=${HOST_PY:-$(command -v python3)}      # needs numpy; captured before ESP-IDF's environment replaces python3
Q=${QEMU_BIN:-$(command -v qemu-system-xtensa || echo "$HOME/esp/qemu/bin/qemu-system-xtensa")}
. "$IDF/export.sh" > /dev/null
mkdir -p "$FW/build_qemu/run" "$LOGS"
# the host C engine's PCM for the blob's self-test sentence (the firmware dumps its own; they must be identical)
make -s -C "$HOST" gen_selftest
case "${VOICE:-female}" in g|G|m|M|male) GREF="$HOST/golden/g/ref0_w8a8.bin" ;; *) GREF="$HOST/golden/ref0_w8a8.bin" ;; esac
"$HOST/gen_selftest" "$W" "$GREF" "$FW/build_qemu/run/selftest_$TAG" 1 8
python3 - <<PY
import sys; sys.path.insert(0, "$IDF/tools")
from idf_py_actions.qemu_ext import QEMU_TARGETS
open("$FW/build_qemu/run/qemu_efuse.bin", "wb").write(QEMU_TARGETS["esp32s3"].default_efuse)
PY
python -m esptool --chip esp32s3 merge_bin --fill-flash-size 16MB -o "$FW/build_qemu/run/flash_$TAG.bin" --flash_mode dio --flash_size 16MB \
  0x0 "$FW/build_qemu/bootloader/bootloader.bin" 0x8000 "$FW/build_qemu/partition_table/partition-table.bin" \
  0x10000 "$FW/build_qemu/itofs.bin" 0x200000 "$W" > /dev/null
PORT=$((5500 + RANDOM % 400))
"$Q" -M esp32s3 -m 8M $QEMU_EXTRA -nographic -monitor none -nic none \
  -drive file="$FW/build_qemu/run/flash_$TAG.bin",if=mtd,format=raw \
  -drive file="$FW/build_qemu/run/qemu_efuse.bin",if=none,format=raw,id=efuse -global driver=nvram.esp32s3.efuse,property=drive,value=efuse \
  -global driver=timer.esp32s3.timg,property=wdt_disable,value=true \
  -serial tcp:127.0.0.1:$PORT,server=on,wait=on > "$FW/build_qemu/run/qemu_$TAG.stdout" 2>&1 &
QPID=$!
echo "qemu pid $QPID port $PORT"
"$HOST_PY" "$ROOT/esp32/tools/qemu/qemu_client.py" 127.0.0.1 $PORT "$LOGS/qemu_$TAG.log" "$FW/build_qemu/run/selftest_$TAG.pcm" || true
kill $QPID 2>/dev/null || true
wait $QPID 2>/dev/null || true
[ -f "$LOGS/qemu_$TAG.log" ] || { echo "QEMU run FAILED: no log"; exit 1; }
grep -E "SELFTEST|PCM COMPARE" "$LOGS/qemu_$TAG.log" || true
if grep -q "PASS" "$LOGS/qemu_$TAG.log" && grep -q "BIT-IDENTICAL" "$LOGS/qemu_$TAG.log" && ! grep -qE "FAIL|DIFFERENT" "$LOGS/qemu_$TAG.log"; then
    echo "QEMU OK ($LOGS/qemu_$TAG.log)"
else
    echo "QEMU check FAILED ($LOGS/qemu_$TAG.log)"; exit 1
fi
