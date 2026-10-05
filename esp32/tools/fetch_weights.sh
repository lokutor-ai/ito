#!/bin/bash
# Put Ito's weights into models/ (the three chip files by default: main int8, main int4, light; pass "all" for the PyTorch file too;
# ONLY=main fetches just the first one).
#   esp32/tools/fetch_weights.sh [all]               female voice (default)
#   VOICE=male esp32/tools/fetch_weights.sh [all]    male voice (VOICE=g also works): ito_male_esp32s3.bin (+ ito_male.pt)
# The weights come from the gated Hugging Face repo lokutor-ai/ito (CC BY-NC-SA 4.0 + models/TERMS.md):
# accept the terms at https://huggingface.co/lokutor-ai/ito, then run `huggingface-cli login` once.
# If you already have the files, set ITO_WEIGHTS_DIR=/path/to/folder and they are copied from there.
# Needs: pip install huggingface_hub
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case "${VOICE:-female}" in
    d|D|f|F|female) CHIP=ito_female_esp32s3.bin; PT=ito_female.pt; EXTRA="ito_female_esp32s3_int4.bin ito_female_esp32s3_light.bin" ;;
    g|G|m|M|male) CHIP=ito_male_esp32s3.bin; PT=ito_male.pt; EXTRA="ito_male_esp32s3_int4.bin ito_male_esp32s3_light.bin" ;;
    *) echo "VOICE must be female or male (or d / g)"; exit 2 ;;
esac
FILES="$CHIP"; [ "$ONLY" != "main" ] && FILES="$CHIP $EXTRA"; [ "$1" = "all" ] && FILES="$FILES $PT"
for f in $FILES; do
    [ -f "$ROOT/models/$f" ] && [ -z "$ITO_WEIGHTS_DIR" ] && { echo "$ROOT/models/$f (already here)"; continue; }
    if [ "$f" = "$CHIP" ] || [ "$f" = "$PT" ]; then
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m ito.weights "$f"
    else      # the optional weight sets (int4 main set, light set): a missing one is not an error, the firmware then has fewer fallbacks
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m ito.weights "$f" || echo "note: $f is not available; the board will run without that fallback set"
    fi
done
