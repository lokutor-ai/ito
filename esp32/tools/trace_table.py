#!/usr/bin/env python3
"""Long-sentence RTF from trace_sim.py's structural weight-DMA model: nothing overlapped / wmode 2 / wmode 3, for the three scenarios
(CPI 1.3 / 1.45 / 1.6, PSRAM 80 / 60 / 40 MB/s), averaged over the full 24-frame chunks of the traced sentences (activation-ring traffic included).
    python3 esp32/tools/trace_table.py <qemu_icprof.log from `icprof 2`> [sentence ...]       (default demo1 demo2; demo2 = the 175-token sentence)
Estimated from exact QEMU instruction counts, not measured on silicon."""
import sys, os, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ts", os.path.join(here, "trace_sim.py")); ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
SC = {"optimistic": (1.30, 80e6), "central": (1.45, 60e6), "pessimistic": (1.60, 40e6)}
POL = (("nothing overlapped", "sum", {}), ("wmode 2", "dma", dict(nbuf=2, look=0)), ("wmode 3", "dma", dict(nbuf=2, look=1)))


def table(log, tag):
    ch = ts.parse(log)[(tag, 1)]
    full = [c for c in sorted(ch) if sum(1 for e in ch[c] if e["kind"] in (0, 1)) >= 120 and c >= 10][:-1]
    out = {}
    for name, (cpi, bw) in SC.items():
        out[name] = []
        for _, pol, kw in POL:
            v = [ts.chunk_time(ts.link(ch[c]), cpi, bw, pol, ts.RING_BPS * 0.3, **kw) / 0.3 for c in full]
            out[name].append(sum(v) / len(v))
    return out, len(full)


if __name__ == "__main__":
    for tag in (sys.argv[2:] or ["demo1", "demo2"]):
        o, n = table(sys.argv[1], tag)
        print("%s (%d full chunks)   %-12s %-12s %-12s" % (tag, n, *SC))
        for i, (lab, _, _) in enumerate(POL):
            print("  %-20s %s" % (lab, "   ".join("%-12.2f" % o[k][i] for k in SC)))
