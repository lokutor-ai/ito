#!/usr/bin/env python3
"""Run texts through the front end of a RUNNING chip / QEMU (serial or TCP) with the `ids <text>` command and compare the ids with the
Python reference (make_reference.py output, or make_stream_reference.py output with --stream).  No audio is made.
   serial_ids.py --port /dev/cu.usbmodemXXXX | --tcp host:port  ref.jsonl [--limit N] [--stream] [--log file] [--out chip_ids.jsonl]
Records the chip's per-text front-end time ("FE utt k: n tokens in X ms")."""
import argparse, json, re, socket, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("--port"); ap.add_argument("--tcp"); ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stream", action="store_true"); ap.add_argument("--log"); ap.add_argument("--out")
ap.add_argument("--wait-ready", type=float, default=600.0); ap.add_argument("--timeout", type=float, default=30.0)
ap.add_argument("--fastfirst0", action="store_true", help="send `fastfirst 0` first (exact say.py utterances)")
a = ap.parse_args()
if a.tcp:
    h, p = a.tcp.split(":"); sock = socket.create_connection((h, int(p))); sock.settimeout(0.5)
    def rd(): 
        try: return sock.recv(65536)
        except socket.timeout: return b""
    def wr(b): sock.sendall(b)
else:
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0.3); ser.dtr = False; ser.rts = False
    def rd(): return ser.read(65536)
    def wr(b): ser.write(b)
logf = open(a.log, "w") if a.log else None
buf = b""
def lines_until(pred, timeout):
    """read until a complete line satisfies pred; returns all lines read"""
    global buf
    out = []; t0 = time.time()
    while time.time() - t0 < timeout:
        d = rd()
        if d:
            buf += d
        while b"\n" in buf:
            l, buf = buf.split(b"\n", 1)
            l = l.decode("utf-8", "replace").rstrip("\r")
            if logf: logf.write(l + "\n"); logf.flush()
            out.append(l)
            if pred(l): return out
    return None
if not lines_until(lambda l: l.startswith("READY"), a.wait_ready):
    sys.exit("no READY from the board")
time.sleep(0.5)
ref = [json.loads(l) for l in open(a.ref)]
if "meta" in ref[0]: ref = ref[1:]
if a.limit: ref = ref[:a.limit]
if a.fastfirst0:
    wr(b"fastfirst 0\n"); lines_until(lambda l: l.startswith("READY"), 10)
ok = bad = fail = 0; ftimes = []; outs = []
for i, r in enumerate(ref):
    text = r["text"]
    if not text.strip(): continue
    wr(("ids " + text + "\n").encode("utf-8"))
    ls = lines_until(lambda l: l.startswith("READY") or l.startswith("ERROR"), a.timeout)
    if ls is None: fail += 1; print("TIMEOUT", i, text[:60]); continue
    utts = []; tms = []
    for l in ls:
        if l.startswith("IDS "): utts.append([int(x) for x in l[4:].split(",")])
        m = re.match(r"FE utt (\d+): (\d+) tokens in ([\d.]+) ms", l)
        if m: tms.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    exp = r["utt"] if a.stream else [r["ids"]]
    exp = [e for e in exp if len(e) <= 400]
    got = [u for u in utts if len(u) <= 400]
    outs.append({"text": text, "utt": utts, "fe": tms})
    if got == exp: ok += 1
    else:
        bad += 1
        if bad <= 10: print("MISMATCH", i, text[:100], [len(x) for x in exp], [len(x) for x in got])
    ftimes += [(n, t) for (_, n, t) in tms]
print("texts", ok + bad + fail, "identical", ok, "different", bad, "no answer", fail)
if ftimes:
    import statistics
    t = [x[1] for x in ftimes]
    print("front-end time per utterance: median %.1f ms, mean %.1f, max %.1f; ids per ms: %.2f" % (statistics.median(t), statistics.mean(t), max(t), sum(x[0] for x in ftimes) / sum(t)))
if a.out: 
    with open(a.out, "w") as f:
        for o in outs: f.write(json.dumps(o, ensure_ascii=False) + "\n")
