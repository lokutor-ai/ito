#!/bin/bash
# Exact instruction counts of the firmware in QEMU (-icount shift=0: 1 instruction = 1 ns of virtual time; CCOUNT ticks at
# 40 MHz there, so 1 tick = 25 instructions, calibrated at run time by ICPROF_CALIB). Builds the profiling firmware
# (build_icprof: -D ITOFS_ICPROF=1, QEMU config) and runs its `icprof` command: the self-test sentence and the 3 demo
# sentences on 1 core, on 2 cores with the GEMM halves serialised (exact per-core split) and on 2 cores in parallel.
#   esp32/tools/qemu/icount_profile.sh /absolute/path/weights.bin        (log in esp32/logs/qemu_icprof.log)
#   python3 esp32/tools/icount_estimate.py esp32/logs/qemu_icprof.log      -> estimated TTFA / RTF (not measured on silicon)
# Several weight sets (flash partitions weights_b / weights_c): W2=/abs/b.bin W3=/abs/c.bin ICPROF_CMD="tier 1;icprof 2" ICPROF_TAG=icprof_set1 esp32/tools/qemu/icount_profile.sh /abs/a.bin
# Start-up schedules and gapless start delay: ICPROF_CMD="icprof 2" esp32/tools/qemu/icount_profile.sh /abs/weights.bin   (every chunk of the 4 sentences
# is traced call by call), then  python3 esp32/tools/sched_eval.py esp32/logs/qemu_icprof.log <sentence>   (needs HOST_PY=<python with numpy> and
# QEMU_BIN=<qemu-system-xtensa> in the environment).
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export HOST_PY=${HOST_PY:-$(command -v python3)}            # the host Python (numpy), before ESP-IDF's environment replaces python3
[ -n "$1" ] && set -- "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"     # the script changes directory: make the weights path absolute
IDF=${IDF_PATH:-$HOME/esp/esp-idf}
. "$IDF/export.sh" > /dev/null
cd "$ROOT/esp32/firmware"
idf.py -B build_icprof -D ITOFS_QEMU=1 -D ITOFS_ICPROF=1 ${ITOFS_EXTRA_DEFS:+-D ITOFS_EXTRA_DEFS="$ITOFS_EXTRA_DEFS"} -D SDKCONFIG=build_icprof/sdkconfig \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu.defaults" set-target esp32s3 > /dev/null
idf.py -B build_icprof -D ITOFS_QEMU=1 -D ITOFS_ICPROF=1 ${ITOFS_EXTRA_DEFS:+-D ITOFS_EXTRA_DEFS="$ITOFS_EXTRA_DEFS"} -D SDKCONFIG=build_icprof/sdkconfig \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu.defaults" build 2>&1 | grep -E "error|Project build complete"
# run_qemu.sh looks in firmware/build_qemu: point it at the profiling build for this run
rm -rf build_qemu.keep; [ -d build_qemu ] && mv build_qemu build_qemu.keep
cp -R build_icprof build_qemu
trap 'rm -rf "$ROOT/esp32/firmware/build_qemu"; [ -d "$ROOT/esp32/firmware/build_qemu.keep" ] && mv "$ROOT/esp32/firmware/build_qemu.keep" "$ROOT/esp32/firmware/build_qemu"' EXIT
QEMU_EXTRA="-icount shift=0" QEMU_SCRIPT="${ICPROF_CMD:-icprof}" "$ROOT/esp32/tools/qemu/run_qemu.sh" "$1" "${ICPROF_TAG:-icprof}" || true
grep -E "ICPROF|Guru|PCM COMPARE|SELFTEST" "$ROOT/esp32/logs/qemu_${ICPROF_TAG:-icprof}.log" | cut -c1-160
