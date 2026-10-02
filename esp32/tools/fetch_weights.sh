#!/bin/bash
# Put Ito's weights into models/ (the chip file by default; pass "all" for the PyTorch file too).
#   esp32/tools/fetch_weights.sh [all]          voice D (female, default)
#   VOICE=g esp32/tools/fetch_weights.sh [all]  voice G (male): ito_v3_G_esp32s3.bin (+ ito_v3_G.pt)
# The weights come from the gated Hugging Face repo lokutor-ai/ito-tts-v3 (CC BY-NC-SA 4.0 + models/TERMS.md):
# accept the terms at https://huggingface.co/lokutor-ai/ito-tts-v3, then run `huggingface-cli login` once.
# If you already have the files, set ITO_WEIGHTS_DIR=/path/to/folder and they are copied from there.
# Needs: pip install huggingface_hub
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case "${VOICE:-d}" in
    d|D) CHIP=ito_v3_esp32s3.bin; PT=ito_v3.pt ;;
    g|G) CHIP=ito_v3_G_esp32s3.bin; PT=ito_v3_G.pt ;;
    *) echo "VOICE must be d or g"; exit 2 ;;
esac
FILES="$CHIP"; [ "$1" = "all" ] && FILES="$CHIP $PT"
for f in $FILES; do
    [ -f "$ROOT/models/$f" ] && [ -z "$ITO_WEIGHTS_DIR" ] && { echo "$ROOT/models/$f (already here)"; continue; }
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m ito.weights "$f"
done
