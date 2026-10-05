"""Drive the QEMU firmware over its TCP serial port: log everything, wait for the boot self-test and demos (READY),
send `say <ids>` (a sentence phonemised by esp32/tools/say.py --dry), `act 16`, `say`, `act 8` and
`test` (dual- and single-core self-test at 8-bit activations, plus the 16-bit record), or the ';'-separated commands in
$QEMU_SCRIPT (the word SAY expands to that say command); compare every PCM the firmware dumps with the host C engine's PCM.
    python3 esp32/tools/qemu/qemu_client.py <host> <port> <log> <host selftest .pcm> [more .pcm: one per later dump]   (run_qemu.sh calls it)"""
import socket, sys, time
import numpy as np

SAY = "0,50,83,54,156,57,135,16,5,16,81,102,61,16,102,68,16,156,51,158,125,57,135,16,3,16,61,58,156,51,158,53,102,112,16,48,123,138,55,16,70,16,62,156,43,102,56,51,16,62,131,156,102,58,16,4"


def main():
    host, port, logp, pcmps = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]      # reference PCM files, one per PCM dump in order (the last repeats)
    ndump = 0
    for _ in range(100):
        try:
            s = socket.create_connection((host, port)); break
        except OSError:
            time.sleep(0.5)
    s.settimeout(1.0)
    log = open(logp, "w")
    buf = b""
    import os
    script = ["say " + SAY, "act 16", "say " + SAY, "act 8", "test"]   # (no `style k`: the shipped blobs have one style and reject it without printing READY)
    if os.environ.get("QEMU_SCRIPT"):             # ';'-separated commands sent one per READY
        script = [c.strip().replace("SAY", "say " + SAY) for c in os.environ["QEMU_SCRIPT"].split(";") if c.strip()]
    step, pcm, t0 = 0, {}, time.time()
    while time.time() - t0 < 6 * 3600:
        try:
            d = s.recv(65536)
            if not d:
                break
            buf += d
        except socket.timeout:
            continue
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            l = line.decode(errors="replace").rstrip("\r")
            if l.startswith("PCM "):
                p = l.split()
                i0 = int(p[1])
                pcm[i0] = [int(h, 16) for h in p[2:]]
                continue
            log.write(f"[{time.time() - t0:8.1f}s] {l}\n"); log.flush()
            if l.startswith("PCMEND"):
                n = int(l.split()[1])
                got = np.array([v - 65536 if v >= 32768 else v for k in sorted(pcm) for v in pcm[k]], np.int16)
                ref = np.fromfile(pcmps[min(ndump, len(pcmps) - 1)], "<i2"); ndump += 1
                same = len(got) == len(ref) and np.array_equal(got, ref)
                nd = int((got[:min(len(got), len(ref))] != ref[:min(len(got), len(ref))]).sum())
                log.write(f"PCM COMPARE vs host C engine: firmware {len(got)} samples (reported {n}), host {len(ref)}, "
                          f"differing samples {nd} -> {'BIT-IDENTICAL' if same else 'DIFFERENT'}\n"); log.flush()
                pcm = {}
            if l.startswith("READY"):
                if step < len(script):
                    s.sendall((script[step] + "\n").encode()); log.write(f">>> {script[step][:60]}\n"); step += 1
                else:
                    log.write("CLIENT DONE\n"); log.close(); return


if __name__ == "__main__":
    main()
