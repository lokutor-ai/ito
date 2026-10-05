#!/bin/bash
# Flash the prebuilt app and the Ito voice model (fetched by fetch_weights.sh; CC BY-NC-SA 4.0 + models/TERMS.md) to an ESP32-S3-DevKitC-1 N16R8 (16 MB flash, 8 MB octal PSRAM).
#   esp32/tools/flash.sh PORT [weights.bin] [app_merged.bin]      female voice (default)
#   VOICE=male esp32/tools/flash.sh PORT                           male voice (VOICE=g also works): ito_male_esp32s3.bin
# A voice has up to three weight sets, flashed to three partitions: the main set (0x200000, int8), the main set with int4 blocks (0x680000) and the
# light set (0xB00000, 4 blocks, int4). At boot the firmware measures them and keeps the best one that runs at a real-time factor <= 0.85
# (the first that does, in that order). Sets that are not in models/ are skipped; the main set alone works as before.
# Needs: pip install esptool. If it fails at 921600 baud, set BAUD=460800. If it does not connect: hold BOOT, tap RST,
# release BOOT, retry. Then open the serial console (115200) and press RST: the board self-tests, benchmarks itself
# (about 1-2 minutes), speaks three demo sentences and prints READY. Please send us the whole log.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT=${1:?usage: flash.sh PORT [weights.bin] [app_merged.bin]}
if [ -z "$2" ]; then "$ROOT/esp32/tools/fetch_weights.sh"; fi      # gated Hugging Face download on first use
case "${VOICE:-female}" in g|G|m|M|male) WDEF=ito_male_esp32s3.bin ;; *) WDEF=ito_female_esp32s3.bin ;; esac
W=${2:-$ROOT/models/$WDEF}; APP=${3:-$ROOT/esp32/prebuilt/ito_app_merged.bin}
EXTRA=""
if [ -z "$2" ]; then      # the default call also flashes the fallback sets of the voice when they are in models/
    B=${WDEF%.bin}
    [ -f "$ROOT/models/${B}_int4.bin" ] && EXTRA="$EXTRA 0x680000 $ROOT/models/${B}_int4.bin"
    [ -f "$ROOT/models/${B}_light.bin" ] && EXTRA="$EXTRA 0xB00000 $ROOT/models/${B}_light.bin"
fi
BAUD=${BAUD:-921600}
if python -m esptool version 2>/dev/null | grep -q "v5"; then WF=write-flash; else WF=write_flash; fi
# erase the two optional partitions first (0x680000 - 0xF80000), so that a set of another voice or an older release cannot linger there
if python -m esptool version 2>/dev/null | grep -q "v5"; then ER=erase-region; else ER=erase_region; fi
python -m esptool --chip esp32s3 -p "$PORT" -b "$BAUD" $ER 0x680000 0x900000
python -m esptool --chip esp32s3 -p "$PORT" -b "$BAUD" $WF 0x0 "$APP" 0x200000 "$W" $EXTRA
echo "weight sets flashed: main$( [ -n "$EXTRA" ] && echo " + fallbacks")"
echo "flashed. Console: python -m serial.tools.miniterm $PORT 115200   then: python esp32/tools/say.py \"Hello!\" --port $PORT"
