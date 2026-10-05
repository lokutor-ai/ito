#!/usr/bin/env python3
"""Structural model of weight-DMA overlap, driven by the exact per-call QEMU -icount trace (ICTRACE lines of `icprof 2`).

Why: icount_estimate.py turns exact instruction counts and PSRAM bytes into time with one knob, `hidden` (the share of PSRAM time
overlapped with compute). QEMU has no GDMA, so that knob cannot be measured here. This script replaces the knob by a small discrete
simulation of what the firmware's weight staging does, event by event (every GEMM call and every row-parallel loop of a chunk, with
its exact per-core instruction count and weight bytes). Policies (chunk_time):

  sum    nothing overlapped: time = compute + bytes / bandwidth                      (the pessimistic definition)
  dma    GDMA tiles: `nbuf` staging buffers per core, the DMA may run `look` GEMM calls ahead of the running one:
           look 0 = wmode 2 (tile i+1 is fetched while tile i is computed; the first tile of every call is waited for),
           look 1 = wmode 3 (the next call's first tiles are fetched during the last tile and the float work in between)
  ideal  max(compute, bytes / bandwidth) for the whole chunk (infinite buffering; unreachable with ~25 KB of free SRAM)

Both cores share ONE PSRAM bus (bandwidth split between the transfers in flight); the activation-ring traffic is added as non-overlapped time.
Everything is estimated, not measured on silicon.
    python3 trace_sim.py <qemu_icprof.log> [sentence=demo1] [schedule id=1]      per-chunk table and the policies side by side
"""
import sys, re, math
CAL = 4000002 / 160305.0
F_HZ = 240e6
TILE = 4096            # staging tile bytes per buffer (kernels_s3.c TILE_BYTES)
TOC = 8
RING_BPS = 4.5e6       # activation-ring PSRAM traffic per audio second (CPU stalls, never overlapped)


def parse(path):
    """{(sentence, schedule id): {chunk: [events]}} from the ICTRACE lines"""
    ev = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"ICTRACE (\w+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m:
            tag, sid, ch, kind, gap, h0, h1, ovh, byt, rows, inn, out = m.groups()
            ev.setdefault((tag, int(sid)), {}).setdefault(int(ch), []).append(
                dict(kind=int(kind), gap=int(gap), h0=int(h0), h1=int(h1), ovh=int(ovh), bytes=int(byt), rows=int(rows), inn=int(inn), out=int(out)))
    return ev


def tiles_for(inn, ch):
    """(bytes, share of the core's compute) per staging tile for a core that owns `ch` output channels of width `inn`."""
    toc = max(1, TILE // inn)
    t, left = [], ch
    while left > 0:
        k = min(toc, left)
        t.append((k * inn, k / ch)); left -= k
    return t


def split(e):
    s = ((e["out"] // 2) + TOC - 1) // TOC * TOC
    return s, e["out"] - s


def chunk_time(evs, cpi, bw, policy, ring_bytes, nbuf=2, look=1):
    """Seconds for one chunk. evs: its events. policy: sum / ideal / dma (nbuf staging buffers per core, `look` = how many GEMM calls
    ahead of the running one the DMA may already fetch: 0 = only the running call (the shipped mode 2), 1 = also the next call)."""
    sec = lambda ticks: ticks * CAL * cpi / F_HZ
    body = [e for e in evs if e["kind"] != 9]
    if policy in ("sum", "ideal"):
        comp = sum(sec(e["gap"] + max(e["h0"], e["h1"]) + e["ovh"]) for e in evs)
        mem = (sum(e["bytes"] for e in body) + ring_bytes) / bw
        return comp + mem if policy == "sum" else max(comp, mem)
    # ---- flat tile lists per core and the sequence of work items
    tiles = [[], []]          # (call index, bytes, compute seconds)
    items = []                # ("cpu", seconds) | ("gemm", call index)
    ncall = 0
    for e in evs:
        if e["gap"]:
            items.append(("cpu", sec(e["gap"])))
        if e["kind"] == 2:
            items.append(("cpu", sec(max(e["h0"], e["h1"]) + e["ovh"])))
        elif e["kind"] in (0, 1):
            chans = [e["out"], 0] if e["kind"] == 0 else list(split(e))
            hs = [e["h0"], e["h1"]] if e["kind"] == 1 else [e["h0"], 0]
            for c in (0, 1):
                if chans[c]:
                    for b, share in tiles_for(e["bytes"] // e["out"] if e["bytes"] and e["out"] else e["inn"], chans[c]):     # bytes per weight row (int4 rows are shorter than `in`)
                        tiles[c].append((ncall, b, sec(hs[c]) * share))
            items.append(("gemm", ncall))
            if e["ovh"]:
                items.append(("cpu", sec(e["ovh"])))
            ncall += 1
    n = [len(tiles[0]), len(tiles[1])]
    fetched = [0, 0]          # tiles fully arrived
    infl = [None, None]       # remaining bytes of the transfer in flight
    comped = [0, 0]           # tiles computed
    t = 0.0
    started = 0               # GEMM calls started
    for it in items:
        if it[0] == "cpu":
            rem = it[1]
            while rem > 1e-12:
                t_, rem = dma_step(t, rem, None, tiles, n, fetched, infl, comped, started, look, nbuf, bw)
                t = t_
            continue
        started += 1
        cur = it[1]
        # compute tiles of this call on both cores
        left = [sum(1 for x in tiles[c][comped[c]:] if x[0] == cur) for c in (0, 1)]
        crem = [None, None]
        while left[0] or left[1] or crem[0] is not None or crem[1] is not None:
            for c in (0, 1):
                if crem[c] is None and left[c] and fetched[c] > comped[c]:
                    crem[c] = tiles[c][comped[c]][2]
            t, _ = dma_step(t, None, crem, tiles, n, fetched, infl, comped, started, look, nbuf, bw, left=left)
    return t + ring_bytes / bw


def dma_step(t, cpu_rem, crem, tiles, n, fetched, infl, comped, started, look, nbuf, bw, left=None):
    """Advance time to the next event; returns (new t, new cpu_rem). cpu_rem: a pure-CPU segment (CPUs busy, DMA runs on);
    crem: per-core tile compute remaining (None = idle/waiting)."""
    for c in (0, 1):                                     # start a transfer where one is allowed
        if infl[c] is None and fetched[c] < n[c]:
            j = fetched[c]
            if j - comped[c] < nbuf and tiles[c][j][0] < started + look:
                infl[c] = float(tiles[c][j][1])
    act = [c for c in (0, 1) if infl[c] is not None]
    rate = bw / len(act) if act else 0.0
    cands = []
    if cpu_rem is not None:
        cands.append(cpu_rem)
    if crem is not None:
        cands += [crem[c] for c in (0, 1) if crem[c] is not None]
    cands += [infl[c] / rate for c in act]
    if not cands:
        raise RuntimeError("deadlock")
    step = max(min(cands), 1e-9)
    for c in act:
        infl[c] -= rate * step
        if infl[c] <= 1e-6:
            infl[c] = None; fetched[c] += 1
    if cpu_rem is not None:
        cpu_rem -= step
    if crem is not None:
        for c in (0, 1):
            if crem[c] is not None:
                crem[c] -= step
                if crem[c] <= 1e-12:
                    crem[c] = None; comped[c] += 1; left[c] -= 1
    return t + step, cpu_rem


def link(evs):
    g = [e for e in evs if e["kind"] in (0, 1)]
    for a, b in zip(g, g[1:]):
        a["_next"] = b
    return evs


def main():
    ev = parse(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "demo1"
    sid = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    SC = {"optimistic": (1.30, 80e6), "central": (1.45, 60e6), "pessimistic": (1.60, 40e6)}
    chunks = ev[(tag, sid)]
    for ch in sorted(chunks):
        evs = link(chunks[ch])
        body = [e for e in evs if e["kind"] != 9]
        byt = sum(e["bytes"] for e in body)
        crit = sum(e["gap"] + max(e["h0"], e["h1"]) + e["ovh"] for e in evs)
        gem = sum(max(e["h0"], e["h1"]) for e in body if e["kind"] in (0, 1))
        print("chunk %d: events %d  GEMM calls %d  weight MB %.2f  crit %.2f Minstr (GEMM compute %.2f)" % (
            ch, len(body), sum(1 for e in body if e["kind"] in (0, 1)), byt / 1e6, crit * CAL / 1e6, gem * CAL / 1e6))
        for name, (cpi, bw) in SC.items():
            res = []
            for pol, kw in (("sum", {}), ("dma", dict(nbuf=2, look=0)), ("dma", dict(nbuf=2, look=1)), ("ideal", {})):
                lab = pol if pol != "dma" else "dma(look %d)" % kw["look"]
                res.append("%s %.0f" % (lab, chunk_time(evs, cpi, bw, pol, 0.0, **kw) * 1e3))
            print("   %-12s ms (no rings)  %s" % (name, "   ".join(res)))


if __name__ == "__main__":
    main()
