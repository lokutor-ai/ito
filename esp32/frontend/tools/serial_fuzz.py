#!/usr/bin/env python3
"""Robustness: send N hostile inputs to a running board over serial (`ids <x>` and `textq <x>`), watch for crash/reset/hang, and check a canary text between them.
   serial_fuzz.py --port P --n 1100 --seed 1 --log fuzz.log [--textq-share 0.15]
Inputs: empty / whitespace, random printable ASCII, random bytes (invalid UTF-8, control characters, NUL), over-long lines (up to 20 kB; the console keeps 4095 bytes),
long words and runs of punctuation, Unicode (emoji, CJK, RTL, combining marks, zero-width, surrogates, overlong UTF-8), numbers, URLs, escape sequences, command look-alikes."""
import argparse, json, random, re, sys, time, serial
ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--n", type=int, default=1100); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--log"); ap.add_argument("--textq-share", type=float, default=0.12); ap.add_argument("--canary-every", type=int, default=20)
a = ap.parse_args()
rnd = random.Random(a.seed)
ser = serial.Serial(a.port, 115200, timeout=0.2); ser.dtr = False; ser.rts = False
logf = open(a.log, "w") if a.log else None
CRASH = re.compile(r"Guru Meditation|abort\(\)|Backtrace|assert failed|==== ItoFS|rst:0x|ets_main|Task watchdog|CPU lockup|Brownout detector|PANIC|ERROR: stack|stack overflow", re.I)
buf = b""; crashes = []; allout = []
def buf_clear():
    global buf
    buf = b""; ser.reset_input_buffer()
def pump():
    global buf
    d = ser.read(65536)
    if d: buf += d
    out = []
    while b"\n" in buf:
        l, buf = buf.split(b"\n", 1); l = l.decode("utf-8", "replace").rstrip("\r"); out.append(l)
        if logf: logf.write(l + "\n")
        if CRASH.search(l) and not l.startswith("SPLIT "): crashes.append(l)
    return out
def drain(quiet=0.8, maxt=6.0):
    """after a command: wait until the board has been silent for `quiet` s (a synthesis prints READY after its MEM line)"""
    t0 = time.time(); last = time.time()
    while time.time() - t0 < maxt and time.time() - last < quiet:
        if pump(): last = time.time()
def run(cmd_bytes, kind, timeout):
    """send; collect until finished. ids: until READY/ERROR. textq: until FE done, then 2.5 s of silence (synthesis ends with MEM line or ERROR)."""
    drain(0.3, 3.0); buf_clear()
    ser.write(cmd_bytes + b"\n"); t0 = time.time(); lines = []; last = time.time(); fe_done = False; n_utt = 0; n_res = 0
    while time.time() - t0 < timeout:
        ls = pump()
        if ls: last = time.time()
        lines += ls
        for l in ls:
            if l.startswith("FE utt"): n_utt += 1
            if l.startswith("FE done"): fe_done = True
            if l.startswith("TIMING") or l.startswith("ERROR itofs") or l.startswith("ERROR no"): n_res += 1
        if kind == "ids" and any(l.startswith("READY") or l.startswith("ERROR") for l in lines): return lines, True
        if kind == "textq" and fe_done and (n_res >= n_utt or time.time() - last > 3.0): drain(); return lines, True
        if kind == "textq" and not fe_done and any(l.startswith("ERROR") for l in lines): return lines, True
        if kind == "other" and time.time() - last > 1.0 and time.time() - t0 > 1.0: return lines, True
    return lines, False
WORDS = "the quick brown fox 12 3.5 $4 Dr. St. e.g. U.S.A. www.x.org a@b.c 1,000 100% 5:30 PM Jan. 2nd #1 Ph.D. etc. o'clock don't it's ...".split()
UNI = ["\U0001F600", "\U0001F4A9", "日本語のテキスト", "中文测试", "مرحبا بالعالم", "שלום עולם", "Привет мир", "ελληνικά", "é̂̃", "a​b‍c", "﻿", "‮", "\ud800".encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass") if False else "x",
       "٣٤٥", "①②③", "½ ¼ ¾", "™ © ®", "—–…", "«»„“”", "ǅ", "ß ẞ", "İ ı", "\u0000".replace("\u0000", "?")]
RAWBAD = [b"\xed\xa0\x80", b"\xc0\xaf", b"\xe0\x80\xaf", b"\xf4\x90\x80\x80", b"\xff\xfe", b"\x80", b"\xc3", b"\xe2\x82", b"\xf0\x9f\x98", b"\x00", b"\x1b[31m", b"\x7f", b"\x03", b"\x04", b"\x08\x08", b"\t\t"]
def gen():
    k = rnd.randrange(16)
    if k == 0: return b""
    if k == 1: return b" " * rnd.randrange(1, 50)
    if k == 2: return bytes(rnd.choice(b"abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,;:!?-'\"()$%&@#/") for _ in range(rnd.randrange(1, 300)))
    if k == 3: return bytes(b for b in (rnd.randrange(1, 256) for _ in range(rnd.randrange(1, 500))) if b not in (10, 13))
    if k == 4:
        n = rnd.choice([4094, 4095, 4096, 4097, 5000, 8000, 20000]); s = " ".join(rnd.choice(WORDS) for _ in range(n // 4)); return s.encode()[:n]
    if k == 5: return (rnd.choice(["a", "9", "é", "-"]) * rnd.choice([100, 500, 1000, 3000])).encode()
    if k == 6: return (rnd.choice([".", "!", "?", ",", ";", "...", "((((", "\"\"\"", "--"]) * rnd.randrange(1, 800)).encode()
    if k == 7: return " ".join(rnd.choice(UNI) for _ in range(rnd.randrange(1, 30))).encode("utf-8")
    if k == 8: return b"".join(rnd.choice(RAWBAD) for _ in range(rnd.randrange(1, 20)))
    if k == 9: return " ".join(rnd.choice(WORDS) for _ in range(rnd.randrange(1, 120))).encode()
    if k == 10: return " ".join(str(rnd.randrange(10 ** rnd.randrange(1, 25))) for _ in range(rnd.randrange(1, 20))).encode()
    if k == 11: return (rnd.choice(["http://", "https://www.", "ftp://", "mailto:"]) + "".join(rnd.choice("abcxyz./-_?=&%20") for _ in range(rnd.randrange(5, 200)))).encode()
    if k == 12: return rnd.choice([b"ids", b"textq", b"textp", b"say 1,2,3", b"tier 99", b"style 99", b"reboot", b"text", b"split", b"first", b"1,2,3,4", b"-1", b"%s%s%n", b"{}", b"\\n\\r"])
    if k == 13: return bytes(rnd.choice(b"0123456789,") for _ in range(rnd.randrange(1, 400)))
    if k == 14: return " ".join(w.upper() if rnd.random() < .5 else w for w in (rnd.choice(WORDS) for _ in range(rnd.randrange(1, 60)))).encode()
    return (("one two three. " * rnd.randrange(1, 40)) + rnd.choice(["", "x" * 500, "…" * 50])).encode()
def sanitize_cmd(cmd):          # no line terminators inside a payload; commands that would play, reboot or take minutes are not sent
    return cmd.replace(b"\n", b" ").replace(b"\r", b" ")
CANARY = (b"ids Hello world, it is 5:30 PM.", "f2f9bb34")
def canary():
    ls, ok = run(CANARY[0], "ids", 15)
    got = [re.search(r"ids fnv32 (\w+)", l) for l in ls if l.startswith("FE utt")]
    return ok and got and got[0] and got[0].group(1) == CANARY[1]
t_start = time.time(); stats = dict(sent=0, ok=0, timeout=0, canary_fail=0, ids=0, textq=0); slow = []
if not canary(): sys.exit("canary failed at the start")
for i in range(a.n):
    payload = sanitize_cmd(gen())
    if payload.startswith((b"ids", b"textq", b"textp", b"say", b"tier", b"style", b"reboot", b"text", b"split", b"first")) and len(payload) < 12:
        cmd, kind = payload, "other"
        if payload.startswith((b"reboot", b"say", b"text ", b"text")) and not payload.startswith((b"textq", b"textp")): cmd = b"help " + payload; kind = "other"   # never reboot / speak
    elif rnd.random() < a.textq_share and len(payload) <= 400:
        cmd, kind = b"textq " + payload, "textq"; stats["textq"] += 1
    else:
        cmd, kind = b"ids " + payload, "ids"; stats["ids"] += 1
    t0 = time.time()
    ls, ok = run(cmd, kind, 90 if kind == "textq" else 40)
    dt = time.time() - t0
    stats["sent"] += 1; stats["ok" if ok else "timeout"] += 1
    if not ok: print("TIMEOUT/HANG input", i, kind, len(cmd), cmd[:60], flush=True)
    if dt > 10: slow.append((i, kind, len(cmd), round(dt, 1)))
    if crashes: print("CRASH LINES after input", i, crashes[:3], flush=True); break
    if (i + 1) % a.canary_every == 0:
        if not canary(): stats["canary_fail"] += 1; print("CANARY FAILED after input", i, flush=True)
        if (i + 1) % 100 == 0: print("progress", i + 1, stats, "%.0f s" % (time.time() - t_start), flush=True)
ok_end = canary()
ser.write(b"split\n"); time.sleep(1.5)
tail = [l for l in pump() if l.startswith("SPLIT")]
print("RESULT", json.dumps(dict(stats, crash_lines=len(crashes), canary_at_end=bool(ok_end), slow=slow[:20], seconds=round(time.time() - t_start))))
print("\n".join(tail))
