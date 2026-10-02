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
import os, re, sys, collections

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
        m = re.search(r"ICPROF (selftest|demo\d) mode (\S+) first (\d+) sched (\d+) tokens (\d+) audio ([\d.]+) chunks (\d+) fnv (\w+) \| first (.*?) \| rest (.*)$", ln)
        if not m:
            continue
        tag, mode, first, sid, tok, aud, nch, fnv, f, r = m.groups()
        rows[(tag, mode, int(sid))] = dict(tokens=int(tok), audio=float(aud), fnv=fnv, first=fields(f), rest=fields(r), firstfr=int(first))
    return rows


def crit(d):  # ticks on the dual-core critical path
    return d["el"] - (d["h0"] + d["h1"] - d["hmax"])


def tm(instr, nbytes, sc):
    cpi, bw, hid = SCEN[sc]
    cpu = instr * cpi / F_HZ
    mem = nbytes / (bw * 1e6)
    return max(cpu + (1 - hid) * mem, mem) * 1e3, cpu * 1e3, mem * 1e3


def chunks_of(path):
    ch = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"ICPROF_CHUNKS (\w+) sched (\d+):(.*)$", ln)
        if m:
            ch[(m.group(1), int(m.group(2)))] = [tuple(float(x) for x in t.split("/")) for t in m.group(3).split()]
    return ch


def main():
    rows = parse(sys.argv[1])
    if not rows:
        sys.exit("no 'ICPROF ... sched <id> ...' lines in %s (logs of the release engine have the older format without 'sched')" % sys.argv[1])
    ch = chunks_of(sys.argv[1])
    out = []
    P = out.append
    sids = sorted({k[2] for k in rows if k[1] == "2core-serial"})
    steady_fr = int(os.environ.get("STEADY", "24"))
    P("EXACT counts (QEMU -icount; 1 CCOUNT tick = %.2f instructions), text-in -> first chunk, per start-up schedule (frames of chunk 0, 1, ...; then %d)" % (CAL, steady_fr))
    P("%-9s %-5s %-6s %-5s | %-10s %-10s | %-8s | %s" % ("sentence", "tok", "sched", "audio", "2-core tot", "crit-path", "PSRAM MB", "bit-exact"))
    for sid in sids:
        for (tag, mode, s2), d in sorted(rows.items()):
            if mode != "2core-serial" or s2 != sid:
                continue
            P("%-9s %-5d s%d f%-2d %-5.1f | %-10.1f %-10.1f | %-8.2f | %s" % (
                tag, d["tokens"], sid, d["firstfr"], d["audio"], d["first"]["el"] * CAL / 1e6, crit(d["first"]) * CAL / 1e6, d["first"]["wb"] / 1e6, d["fnv"]))
    P("")
    d1 = rows.get(("demo1", "2core-serial", sids[0]))
    P("Per core in the first chunk (serialised halves): core0 %.1f M, core1 %.1f M instr (demo1, sched %d)" % (d1["first"]["h0"] * CAL / 1e6, d1["first"]["h1"] * CAL / 1e6, sids[0]))
    P("")
    P("ESTIMATED time-to-first-audio (ms) -- estimated from exact QEMU instruction counts, not measured on silicon")
    P("%-9s %-5s %-6s | %-22s %-22s %-22s" % ("sentence", "tok", "first", "optimistic (cpu/mem)", "central", "pessimistic"))
    for sid in sids:
        for tag in ["selftest", "demo0", "demo1", "demo2"]:
            d = rows.get((tag, "2core-serial", sid))
            if not d:
                continue
            ins, byt = crit(d["first"]) * CAL, d["first"]["wb"]
            cells = []
            for sc in SCEN:
                t, c, mm = tm(ins, byt, sc)
                cells.append("%5.0f (%4.0f/%4.0f)       " % (t, c, mm))
            P("%-9s %-5d s%d f%-2d | %s" % (tag, d["tokens"], sid, d["firstfr"], " ".join(cells)))
    P("  (add <= 1 DMA descriptor (<= 20 ms) before the first sample reaches the DAC)")
    P("")
    P("ESTIMATED steady-state RTF: all audio after the first chunk, INCLUDING the start-up ramp chunks (< 1 needed)")
    P("%-9s %-5s %-6s | %-8s %-9s %-9s | %-12s %-12s %-12s" % ("sentence", "tok", "sched", "crit M/s", "PSRAM MB/s", "(+rings)", "optimistic", "central", "pessimistic"))
    for sid in sids:
        for tag in ["selftest", "demo0", "demo1", "demo2"]:
            d = rows.get((tag, "2core-serial", sid))
            if not d:
                continue
            aud = d["audio"] - d["firstfr"] * HOP / SR
            ins, byt = crit(d["rest"]) * CAL, d["rest"]["wb"] + RING_BPS * aud
            cells = []
            for sc in SCEN:
                t, c, mm = tm(ins, byt, sc)
                cells.append("%5.2f(c%.2f m%.2f)" % (t / 1e3 / aud, c / 1e3 / aud, mm / 1e3 / aud))
            P("%-9s %-5d s%d     | %-8.0f %-9.1f %-9.1f | %s" % (tag, d["tokens"], sid, ins / aud / 1e6, d["rest"]["wb"] / aud / 1e6, byt / aud / 1e6, " ".join(cells)))
    P("")
    P("ESTIMATED RTF of the steady-state chunks alone (full %d-frame chunks only: the long-sentence limit)" % steady_fr)
    P("%-9s %-5s %-6s | %-8s %-9s %-9s | %-12s %-12s %-12s" % ("sentence", "tok", "sched", "crit M/s", "PSRAM MB/s", "(+rings)", "optimistic", "central", "pessimistic"))
    for sid in sids:
        for tag in ["selftest", "demo0", "demo1", "demo2"]:
            L = [c for c in ch.get((tag, sid), []) if c[0] == steady_fr * HOP]
            if not L:
                continue
            aud = sum(c[0] for c in L) / SR
            ins, wb = sum(c[1] for c in L) * CAL, sum(c[2] for c in L)
            byt = wb + RING_BPS * aud
            cells = []
            for sc in SCEN:
                t, c, mm = tm(ins, byt, sc)
                cells.append("%5.2f(c%.2f m%.2f)" % (t / 1e3 / aud, c / 1e3 / aud, mm / 1e3 / aud))
            P("%-9s %-5d s%d     | %-8.0f %-9.1f %-9.1f | %s" % (tag, rows[(tag, "2core-serial", sid)]["tokens"], sid, ins / aud / 1e6, wb / aud / 1e6, byt / aud / 1e6, " ".join(cells)))
    P("  sensitivity: at 27 MB/s (forum memcpy EXT->EXT) the PSRAM term alone is %.2f of real time (demo1, sched %d)" % (
        rows[("demo1", "2core-serial", sids[0])]["rest"]["wb"] / (rows[("demo1", "2core-serial", sids[0])]["audio"]) / 27e6, sids[0]))
    # ---- sensitivity grids (central CPI) for the first schedule
    sid0 = int(os.environ.get("SENS_SCHED", str(sids[0])))
    d = rows[("demo1", "2core-serial", sid0)]
    P("")
    P("Sensitivity (CPI 1.45, demo1, sched %d): TTFA ms of the first chunk / steady RTF, PSRAM MB/s x share of PSRAM time hidden by overlap" % sid0)
    P("%-10s" % "MB/s" + "".join("hidden %-4.2f      " % h for h in (0.0, 0.5, 0.8, 1.0)))
    a2 = d["audio"] - d["firstfr"] * HOP / SR
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
    P("S = max over chunks k of (ready_k - audio_before_k); I2S descriptor not included; 'lag' = how far synthesis is behind playback at the end")
    P("%-9s %-5s %-6s | %-24s %-24s %-24s" % ("sentence", "tok", "sched", "optimistic", "central", "pessimistic"))
    for sid in sids:
        for tag in ["demo0", "demo1", "demo2"]:
            L = ch.get((tag, sid))
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
            P("%-9s %-5d s%d     | %s" % (tag, rows[(tag, "2core-serial", sid)]["tokens"], sid, " ".join(cells)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
