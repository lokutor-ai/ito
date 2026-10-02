#!/bin/bash
# Flash the prebuilt app and the Ito voice model (fetched by fetch_weights.sh; CC BY-NC-SA 4.0 + models/TERMS.md) to an ESP32-S3-DevKitC-1 N16R8 (16 MB flash, 8 MB octal PSRAM).
#   esp32/tools/flash.sh PORT [weights.bin] [app_merged.bin]      female voice (default)
#   VOICE=male esp32/tools/flash.sh PORT                           male voice (VOICE=g also works): ito_male_esp32s3.bin
# Needs: pip install esptool. If it fails at 921600 baud, set BAUD=460800. If it does not connect: hold BOOT, tap RST,
# release BOOT, retry. Then open the serial console (115200) and press RST: the board self-tests, benchmarks itself
# (about 1-2 minutes), speaks three demo sentences and prints READY. Please send us the whole log.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT=${1:?usage: flash.sh PORT [weights.bin] [app_merged.bin]}
if [ -z "$2" ]; then "$ROOT/esp32/tools/fetch_weights.sh"; fi      # gated Hugging Face download on first use
case "${VOICE:-female}" in g|G|m|M|male) WDEF=ito_male_esp32s3.bin ;; *) WDEF=ito_female_esp32s3.bin ;; esac
W=${2:-$ROOT/models/$WDEF}; APP=${3:-$ROOT/esp32/prebuilt/ito_app_merged.bin}
BAUD=${BAUD:-921600}
if python -m esptool version 2>/dev/null | grep -q "v5"; then WF=write-flash; else WF=write_flash; fi
python -m esptool --chip esp32s3 -p "$PORT" -b "$BAUD" $WF 0x0 "$APP" 0x200000 "$W"
echo "flashed. Console: python -m serial.tools.miniterm $PORT 115200   then: python esp32/tools/say.py \"Hello!\" --port $PORT"
