#!/usr/bin/env python3
"""Estimated ESP32-S3 TTFA / RTF from exact QEMU -icount instruction counts.   python3 esp32/tools/icount_estimate.py esp32/logs/qemu_icprof.log

Input: the ICPROF lines printed by the `icprof` firmware command (tools/qemu/icount_profile.sh). Everything measured is
EXACT (QEMU counts instructions, not time): per-core instructions from text-in to the first chunk, per audio second,
the dual-core critical path, and the bytes of weights the GEMM staging reads (PSRAM traffic). Time is then ESTIMATED:

  cpu_ms = critical-path instructions * CPI / 240 MHz           (CPI 1.3 / 1.45 / 1.6, the Oido range)
  mem_ms = weight bytes (+ activation-ring bytes) / PSRAM bandwidth          (80 / 60 / 40 MB/s effective)
  time   = max(cpu + (1 - hidden) * mem, mem)                   (hidden = share of PSRAM time overlapped with compute
                                                                 by the other core / staging: 1.0 / 0.5 / 0.0)

The PSRAM bus is shared by both cores, so mem uses the bytes of BOTH cores. Not measured on silicon.
"""
import re, sys, collections

CAL = 4000002 / 160305.0         # instructions per CCOUNT tick under -icount shift=0 (ICPROF_CALIB); ~24.95
F_HZ = 240e6
HOP, SR = 300, 24000
SCEN = {  # name: (CPI, PSRAM MB/s, hidden share)
    "optimistic":  (1.30, 80.0, 1.0),
    "central":     (1.45, 60.0, 0.5),
    "pessimistic": (1.60, 40.0, 0.0),
}
RING_BPS = 4.5e6                  # activation-ring PSRAM traffic per audio second, written + ~2x read (1.49 MB/s written, from the stage-ring widths)


def fields(seg):
    return {k: float(v) for k, v in re.findall(r"(\w+) (-?[\d.]+)", seg)}


def parse(path):
    rows = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"ICPROF (selftest|demo\d) mode (\S+)(?: first (\d+))? tokens (\d+) audio ([\d.]+) chunks (\d+) fnv (\w+) \| first (.*?) \| rest (.*)$", ln)
        if not m:
            continue
        tag, mode, first, tok, aud, nch, fnv, f, r = m.groups()
        rows[(tag, mode, int(first or 8))] = dict(tokens=int(tok), audio=float(aud), fnv=fnv, first=fields(f), rest=fields(r))
    return rows


def crit(d):  # ticks on the dual-core critical path
    return d["el"] - (d["h0"] + d["h1"] - d["hmax"])


def tm(instr, nbytes, sc):
    cpi, bw, hid = SCEN[sc]
    cpu = instr * cpi / F_HZ
    mem = nbytes / (bw * 1e6)
    return max(cpu + (1 - hid) * mem, mem) * 1e3, cpu * 1e3, mem * 1e3


def main():
    rows = parse(sys.argv[1])
    out = []
    P = out.append
    firsts = sorted({k[2] for k in rows if k[1] == "2core-serial"})
    P("EXACT counts (QEMU -icount; 1 CCOUNT tick = %.2f instructions), text-in -> first chunk" % CAL)
    P("%-9s %-5s %-6s %-5s | %-10s %-10s %-10s | %-8s | %s" % ("sentence", "tok", "first", "audio", "1-core M", "2-core tot", "crit-path", "PSRAM MB", "bit-exact"))
    for fr in firsts:
        for (tag, mode, f), d in sorted(rows.items()):
            if mode != "2core-serial" or f != fr:
                continue
            one = rows.get((tag, "1core", f))
            P("%-9s %-5d %-2d fr  %-5.1f | %-10.1f %-10.1f %-10.1f | %-8.2f | %s" % (
                tag, d["tokens"], f, d["audio"], (one["first"]["el"] * CAL / 1e6) if one else float("nan"),
                d["first"]["el"] * CAL / 1e6, crit(d["first"]) * CAL / 1e6, d["first"]["wb"] / 1e6, d["fnv"]))
    P("")
    P("Per core in the first chunk (serialised halves): core0 %.1f M, core1 %.1f M instr (demo1, first=%d)" % (
        *(lambda d: (d["first"]["h0"] * CAL / 1e6, d["first"]["h1"] * CAL / 1e6))(rows[("demo1", "2core-serial", firsts[0])]), firsts[0]))
    P("")
    P("ESTIMATED time-to-first-audio (ms) -- estimated from exact QEMU instruction counts, not measured on silicon")
    P("%-9s %-5s %-6s | %-22s %-22s %-22s" % ("sentence", "tok", "first", "optimistic (cpu/mem)", "central", "pessimistic"))
    res = collections.defaultdict(dict)
    for fr in firsts:
        for tag in ["selftest", "demo0", "demo1", "demo2"]:
            d = rows.get((tag, "2core-serial", fr))
            if not d:
                continue
            ins, byt = crit(d["first"]) * CAL, d["first"]["wb"]
            cells = []
            for sc in SCEN:
                t, c, mm = tm(ins, byt, sc)
                res[(tag, fr)][sc] = t
                cells.append("%5.0f (%4.0f/%4.0f)       " % (t, c, mm))
            P("%-9s %-5d %-2d fr   | %s" % (tag, d["tokens"], fr, " ".join(cells)))
    P("  (add <= 1 DMA descriptor (<= 20 ms) before the first sample reaches the DAC)")
    P("")
    P("ESTIMATED steady-state RTF (all audio after the first chunk; < 1 needed)")
    P("%-9s %-5s %-6s | %-8s %-9s %-9s | %-12s %-12s %-12s" % ("sentence", "tok", "first", "crit M/s", "PSRAM MB/s", "(+rings)", "optimistic", "central", "pessimistic"))
    for fr in firsts[:1] if False else [firsts[-1]]:
        for tag in ["selftest", "demo0", "demo1", "demo2"]:
            d = rows.get((tag, "2core-serial", fr))
            if not d:
                continue
            aud = d["audio"] - fr * HOP / SR
            ins, byt = crit(d["rest"]) * CAL, d["rest"]["wb"] + RING_BPS * aud
            cells = []
            for sc in SCEN:
                t, c, mm = tm(ins, byt, sc)
                cells.append("%5.2f(c%.2f m%.2f)" % (t / 1e3 / aud, c / 1e3 / aud, mm / 1e3 / aud))
            P("%-9s %-5d %-2d fr   | %-8.0f %-9.1f %-9.1f | %s" % (tag, d["tokens"], fr, ins / aud / 1e6, d["rest"]["wb"] / aud / 1e6, byt / aud / 1e6, " ".join(cells)))
    P("  sensitivity: at 27 MB/s (forum memcpy EXT->EXT) the PSRAM term alone is %.2f of real time" % (
        rows[("demo1", "2core-serial", firsts[-1])]["rest"]["wb"] / (rows[("demo1", "2core-serial", firsts[-1])]["audio"]) / 27e6))
    # ---- sensitivity grids (central CPI) and gapless start, from the per-chunk counts ----
    if ("demo1", "2core-serial", 2) not in rows:       # older logs (release engine, 8-frame first chunk only)
        print("\n".join(out)); return
    d = rows[("demo1", "2core-serial", 2)]
    P("")
    P("Sensitivity (CPI 1.45, demo1): TTFA ms of the 2-frame first chunk / steady RTF, PSRAM MB/s x share of PSRAM time hidden by overlap")
    P("%-10s" % "MB/s" + "".join("hidden %-4.2f      " % h for h in (0.0, 0.5, 0.8, 1.0)))
    a2 = d["audio"] - 2 * HOP / SR
    for bw in (27, 40, 50, 60, 80):
        cells = []
        for h in (0.0, 0.5, 0.8, 1.0):
            SCEN["_s"] = (1.45, bw, h)
            t1 = tm(crit(d["first"]) * CAL, d["first"]["wb"], "_s")[0]
            tr = tm(crit(d["rest"]) * CAL, d["rest"]["wb"] + RING_BPS * a2, "_s")[0] / 1e3 / a2
            cells.append("%4.0f ms / %4.2f   " % (t1, tr))
        P("%-10d" % bw + "".join(cells))
    del SCEN["_s"]
    P("")
    P("Gapless start (audio never underruns): smallest delay S after text-in so that every later chunk is ready when the speaker needs it")
    P("S = max over chunks k of (ready_k - audio_before_k); first chunk 2 frames then 8-frame chunks; I2S descriptor not included")
    ch = {}
    for ln in open(sys.argv[1], errors="replace"):
        m = re.search(r"ICPROF_CHUNKS (\w+) first (\d+):(.*)$", ln)
        if m:
            ch[(m.group(1), int(m.group(2)))] = [tuple(float(x) for x in t.split("/")) for t in m.group(3).split()]
    P("%-9s %-5s %-6s | %-24s %-24s %-24s" % ("sentence", "tok", "first", "optimistic", "central", "pessimistic"))
    for fr in (2, 8):
        for tag in ["demo0", "demo1", "demo2"]:
            L = ch.get((tag, fr))
            if not L:
                continue
            cells = []
            for sc in SCEN:
                t, S, aud = 0.0, 0.0, 0.0
                for smp, tk, wb, _ in L:
                    a = smp / SR
                    t += tm(tk * CAL, wb + RING_BPS * a, sc)[0] / 1e3
                    S = max(S, t - aud)
                    aud += a
                cells.append("%5.0f ms (last chunk lag %4.0f)" % (S * 1e3, (t - aud) * 1e3))
            P("%-9s %-5d %-2d fr   | %s" % (tag, rows[(tag, "2core-serial", fr)]["tokens"], fr, " ".join(cells)))
    P("  (a negative-free S above RTF-limited sentences grows with length when RTF > 1: lag at the end shows it)")
    print("\n".join(out))


if __name__ == "__main__":
    main()
