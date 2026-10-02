#!/usr/bin/env python3
"""Gapless start delay S, time to first audio and RTF of start-up schedules, from a `icprof 2` log (ICTRACE per chunk + ICPROF_CHUNKS).
Time per chunk = trace_sim.chunk_time (structural weight-DMA model, see trace_sim.py) for each policy:
  sum  nothing overlapped (the pessimistic definition)      dma0  the shipped GDMA mode (no cross-call prefetch)
  dma1 cross-call prefetch (mode 3)                          knob  icount_estimate.py's hidden-share formula (1 / 0.5 / 0 by scenario)
    python3 sched_eval.py log [demo1|demo0]
Estimated from exact QEMU instruction counts, not measured on silicon."""
import sys, re, importlib.util, os
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ts", os.path.join(here, "trace_sim.py")); ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
CAL, SR, HOP = ts.CAL, 24000, 300
SC = {"optimistic": (1.30, 80e6, 1.0), "central": (1.45, 60e6, 0.5), "pessimistic": (1.60, 40e6, 0.0)}

def load(path):
    tr, ch, fnv = {}, {}, {}
    for ln in open(path, errors="replace"):
        m = re.search(r"ICTRACE (\w+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m:
            tag, sid, c, kind, gap, h0, h1, ovh, byt, rows, inn, out = m.groups()
            tr.setdefault((tag, int(sid)), {}).setdefault(int(c), []).append(dict(kind=int(kind), gap=int(gap), h0=int(h0), h1=int(h1), ovh=int(ovh), bytes=int(byt), rows=int(rows), inn=int(inn), out=int(out)))
            continue
        m = re.search(r"ICPROF_CHUNKS (\w+) sched (\d+):(.*)$", ln)
        if m:
            ch[(m.group(1), int(m.group(2)))] = [tuple(float(x) for x in t.split("/")) for t in m.group(3).split()]
        m = re.search(r"ICPROF (\w+) mode 2core-serial first \d+ sched (\d+) .* fnv (\w+)", ln)
        if m:
            fnv[(m.group(1), int(m.group(2)))] = m.group(3)
    return tr, ch, fnv

def times(tr, ch, tag, sid, sc, pol):
    cpi, bw, hid = SC[sc]
    out = []
    chunks = tr[(tag, sid)]
    for k in sorted(chunks):
        evs = ts.link(chunks[k])
        smp = ch[(tag, sid)][k][0] if k < len(ch[(tag, sid)]) else None
        if smp is None:
            continue
        ring = ts.RING_BPS * smp / SR
        if pol == "knob":
            body = [e for e in evs if e["kind"] != 9]
            ins = sum(e["gap"] + max(e["h0"], e["h1"]) + e["ovh"] for e in evs) * CAL
            mem = (sum(e["bytes"] for e in body) + ring) / bw
            cpu = ins * cpi / ts.F_HZ
            t = max(cpu + (1 - hid) * mem, mem)
        elif pol == "sum":
            t = ts.chunk_time(evs, cpi, bw, "sum", ring)
        elif pol == "dma0":
            t = ts.chunk_time(evs, cpi, bw, "dma", ring, nbuf=2, look=0)
        else:
            t = ts.chunk_time(evs, cpi, bw, "dma", ring, nbuf=2, look=1)
        out.append((smp / SR, t))
    return out

def gapless(tl):
    t, aud, S = 0.0, 0.0, 0.0
    for a, c in tl:
        t += c
        S = max(S, t - aud)
        aud += a
    return tl[0][1], S, t / aud if aud else 0

def main():
    tr, ch, fnv = load(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "demo1"
    pols = sys.argv[3].split(",") if len(sys.argv) > 3 else ["knob", "dma0", "dma1", "sum"]
    sids = sorted(s for (t, s) in tr if t == tag)
    print("bit-identical audio across schedules (fnv per sentence):", sorted(set(v for (t, s), v in fnv.items() if t == tag)))
    for pol in pols:
        print("\\npolicy %s   (TTFA ms / gapless start delay S ms / whole-sentence RTF)" % pol)
        print("%-34s" % "schedule (frames of chunk 0,1,..; then 24)" + "".join("%-30s" % s for s in SC))
        for sid in sids:
            sm = [round(c[0] / HOP) for c in ch[(tag, sid)][:7]]
            lab = ",".join(str(x) for x in sm[:7])
            cells = []
            for sc in SC:
                tl = times(tr, ch, tag, sid, sc, pol)
                f, S, r = gapless(tl)
                cells.append("%4.0f / %5.0f / %.2f" % (f * 1e3, S * 1e3, r))
            print("s%-2d %-30s" % (sid, lab) + "".join("%-30s" % c for c in cells))

if __name__ == "__main__":
    main()


def report(path, sid, pols=("knob", "dma0", "dma1", "sum")):
    """one schedule across every sentence in the log: TTFA / S / RTF per scenario and policy"""
    tr, ch, fnv = load(path)
    tags = sorted(t for (t, s) in tr if s == sid)
    for pol in pols:
        print("policy %s" % pol)
        for tag in tags:
            cells = []
            for sc in SC:
                f, S, r = gapless(times(tr, ch, tag, sid, sc, pol))
                cells.append("%4.0f / %5.0f / %.2f" % (f * 1e3, S * 1e3, r))
            print("  %-9s fnv %-9s %s" % (tag, fnv.get((tag, sid), "?"), "   ".join(cells)))
