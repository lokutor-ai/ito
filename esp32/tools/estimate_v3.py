"""TTFA / RTF estimates for the v3 engine on the ESP32-S3 from the host op profile (engine/opcount_v3.c logs).

    python3 esp32/tools/estimate_v3.py results/opcount.log        (written by esp32/host/opcount_v3)

Model (stated, not measured): time = executed int8 GEMM MACs / G + non-GEMM float work / 240 MHz, where G is the
EFFECTIVE int8 GEMM throughput on the board (it includes the PSRAM weight traffic: 38 MB of weights per second of audio
at the 8-frame chunk) and the float work is opcount's cycle estimate (it runs on core 0 only, serially with the GEMMs).
The board benchmark (BENCH_TTS lines: "GEMM ... ms = x GMAC/s, non-GEMM ... ms") measures both terms directly."""
import os, re, sys
import numpy as np

GS = (0.3, 0.5, 1.0, 2.0)
CPU_HZ = 240e6


def parse(path):
    rows, per_s = [], {}
    for line in open(path):
        m = re.search(r"golden \d+: (\d+) tokens, ([\d.]+) s \| before the first chunk: ([\d.]+) M int8 MACs, ([\d.]+) M f32 MACs, "
                      r"non-GEMM ~([\d.]+) M cycles, ([\d.]+) MB", line)
        if m:
            rows.append([float(x) for x in m.groups()])
        m = re.search(r"int8 GEMM MACs executed\s+([\d.]+) M", line)
        if m: per_s["int8"] = float(m.group(1)) * 1e6
        m = re.search(r"f32 MACs \(non-GEMM\)\s+([\d.]+) M", line)
        if m: per_s["f32"] = float(m.group(1)) * 1e6
        m = re.search(r"weights read by the GEMM\s+([\d.]+) MB", line)
        if m: per_s["wmb"] = float(m.group(1))
        m = re.search(r"ESTIMATED non-GEMM work: ([\d.]+) M cycles", line)
        if m: per_s["cyc"] = float(m.group(1)) * 1e6
    return np.array(rows), per_s


def main():
    R, P = parse(sys.argv[1])
    n, i8, cy, wmb = R[:, 0], R[:, 2] * 1e6, R[:, 4] * 1e6, R[:, 5]
    a8, b8 = np.polyfit(n, i8, 1)[::-1]
    ac, bc = np.polyfit(n, cy, 1)[::-1]
    flat = abs(b8) < 1e3 and abs(bc) < 1e3          # work before the first chunk does not grow with sentence length
    label = os.environ.get("CHAIN", "forward-GRU front, fixed style (incremental text side)" if flat
                           else "front with a whole-sentence text side")
    print(f"{label}, fitted on {len(n)} golden sentences:")
    print(f"  int8 MACs before the first chunk = {a8 / 1e6:.1f} M + {b8 / 1e6:.3f} M/token; non-GEMM ~{ac / 1e6:.1f} M + {bc / 1e6:.3f} M cycles/token; "
          f"weights read {wmb.min():.1f}-{wmb.max():.1f} MB")
    print(f"  per second of audio: {P['int8'] / 1e6:.1f} M int8 MACs, {P['f32'] / 1e6:.2f} M f32 MACs, ~{P['cyc'] / 1e6:.1f} M non-GEMM cycles "
          f"({P['cyc'] / CPU_HZ:.2f} core), {P['wmb']:.1f} MB weights")
    print("\nTTFA (first 100 ms chunk ready), ms = int8 / G + non-GEMM / 240 MHz:")
    print("| tokens | int8 ops | non-GEMM | " + " | ".join(f"G = {g} GOPS" for g in GS) + " | G needed for < 200 ms |")
    print("|---|---|---|" + "---|" * len(GS) + "---|")
    for tok, lab in ((25, "25 (short)"), (87, "87 (val p25)"), (110, "110 (val median)"), (132, "132 (val p75)"), (165, "165 (val p95)"),
                     (182, "182 (val max, long)")):
        o = a8 + b8 * tok; f = (ac + bc * tok) / CPU_HZ
        t = [o / g / 1e9 + f for g in GS]
        need = o / 1e9 / (0.2 - f) if f < 0.2 else float("inf")
        print(f"| {lab} | {o / 1e6:.0f} M | {f * 1e3:.0f} ms | " + " | ".join(f"{x * 1e3:.0f}" for x in t) +
              f" | {need:.2f} GOPS |" if np.isfinite(need) else " | impossible |")
    g2 = 2.0
    if flat:
        print("  the work before the first chunk is the same for any sentence length")
    else:
        nmax = (0.2 - ac / CPU_HZ - a8 / g2 / 1e9) / (b8 / g2 / 1e9 + bc / CPU_HZ)
        print(f"  at {g2} GOPS TTFA < 200 ms holds up to ~{nmax:.0f} tokens")
    if len(sys.argv) > 2:
        F, PF = parse(sys.argv[2])
        o, f = F[:, 2].mean() * 1e6, F[:, 4].mean() * 1e6 / CPU_HZ
        print(f"\nsecond profile, any sentence length: {o / 1e6:.1f} M int8 + "
              f"{F[:, 3].mean():.2f} M f32 MACs, non-GEMM ~{f * 1e3:.0f} ms, {F[:, 5].mean():.1f} MB weights before the first chunk")
        print("| | " + " | ".join(f"G = {g}" for g in GS) + " | G needed for < 200 ms |")
        print("|---|" + "---|" * len(GS) + "---|")
        print("| TTFA ms | " + " | ".join(f"{(o / g / 1e9 + f) * 1e3:.0f}" for g in GS) + f" | {o / 1e9 / (0.2 - f):.2f} GOPS |")
    rt = [P["int8"] / g / 1e9 + P["cyc"] / CPU_HZ for g in GS]
    print("\nRTF = int8 per second / G + non-GEMM per second / 240 MHz:")
    print("| | " + " | ".join(f"G = {g}" for g in GS) + " | G needed for RTF < 1 |")
    print("|---|" + "---|" * len(GS) + "---|")
    print("| RTF | " + " | ".join(f"{x:.2f}" for x in rt) + f" | {P['int8'] / 1e9 / (1 - P['cyc'] / CPU_HZ):.2f} GOPS |")
    print(f"\nPSRAM: the GEMMs read {P['wmb']:.1f} MB of weights per second of audio, i.e. {P['int8'] / P['wmb'] / 1e6:.1f} MACs per weight byte; "
          f"G is capped at {P['int8'] / P['wmb'] / 1e6:.1f} x (effective PSRAM MB/s) / 1000 GOPS (40 MB/s -> "
          f"{P['int8'] / P['wmb'] / 1e6 * 0.04:.2f}, 80 MB/s -> {P['int8'] / P['wmb'] / 1e6 * 0.08:.2f}) unless the copy overlaps compute (wmode gdma).")


if __name__ == "__main__":
    main()
