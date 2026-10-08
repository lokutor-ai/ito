"""Score every system in $BENCH_RAW (default ./raw) on the same post-processed audio; write results.json, results_per_clip
(work/per_clip.json) and the tables (work/results_table.md, pasted into results.md).

Post-processing (identical for every system): mono -> 24 kHz (soxr) -> trim leading/trailing silence (librosa top_db 40,
then 50 ms padding) -> loudness -20 LUFS (BS.1770, pyloudnorm) -> peak-limit -1 dBFS -> 16-bit. Showcase prompts are
copied to audio/<system>/<id>.wav. (The raw chip-exact PCM stays in raw/ito_v3_chip/.)

Metrics per clip:
  utmosv2      UTMOSv2 (sarulab-speech/UTMOSv2, pretrained)        naturalness MOS predictor, 1-5
  utmos22      UTMOS22 strong (tarepan/SpeechMOS v1.2.0)            older predictor, kept for continuity with Ito logs
  dnsmos_*     DNSMOS P.835 SIG/BAK/OVRL (Microsoft DNS-Challenge sig_bak_ovr.onnx), speaker-independent signal quality
  wer / cer    Whisper large-v3 (HF transformers fp16, beam 5, no sampling, language en; all clips < 30 s); Whisper EnglishTextNormalizer + two rules
               (digit.digit / digit:digit -> two numbers; interjection spellings dropped) on both reference and
               hypothesis; corpus-level (total edits / total reference words or chars)
  f0_std_st    std of voiced log-F0 in semitones (pYIN 50-600 Hz, 10 ms hop)   prosody-variability proxy
  f0_range_st  5th-95th percentile voiced F0 span in semitones
  dur_s        trimmed clip duration
Statistics: mean over prompts with a 95% percentile bootstrap CI (10,000 resamples of prompts, seed 0); WER/CER CIs
resample prompts and recompute the corpus ratio. Paired deltas vs the reference system (default ito_v3_chip) resample
prompts jointly (reference: ito_192_female, the headline row).

  python score.py [--systems a,b,...] [--skip_utmos22] [--report_only]
Runs on one GPU (Whisper large-v3 fp16 ~4 GB, UTMOSv2 ~2 GB); pYIN runs on CPU workers."""
import os, sys, json, glob, argparse, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from benchlib import prompts

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.environ.get("BENCH_RAW", os.path.join(HERE, "raw")))
ap.add_argument("--work", default=os.path.join(HERE, "work"))
ap.add_argument("--systems", default="")
ap.add_argument("--ref_system", default="ito_192_female")
ap.add_argument("--asr", default="large-v3")
ap.add_argument("--dnsmos", default=os.path.expanduser("~/bench_envs/models/dnsmos/sig_bak_ovr.onnx"))
ap.add_argument("--skip_utmos22", action="store_true")
ap.add_argument("--report_only", action="store_true")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
SR = 24000
DEV = os.environ.get("BENCH_DEVICE", "")   # "", cuda, mps or cpu ("" = cuda if available, else cpu)
P = prompts()
PID = [p["id"] for p in P]
TEXT = {p["id"]: p["text"] for p in P}
systems = a.systems.split(",") if a.systems else sorted(d for d in os.listdir(a.raw)
                                                       if not d.startswith("_") and os.path.isdir(os.path.join(a.raw, d)))
norm_dir = os.path.join(a.work, "norm")
per_path = os.path.join(a.work, "per_clip.json")
PER = json.load(open(per_path)) if os.path.exists(per_path) else {}


def save_per():
    json.dump(PER, open(per_path, "w"), indent=0)


# ---------------------------------------------------------------- normalisation
def normalise(src, dst):
    import soundfile as sf, librosa, pyloudnorm as pyln
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    _, (b, e) = librosa.effects.trim(y, top_db=40, frame_length=1024, hop_length=120)
    pad = int(0.05 * SR)
    y = y[max(0, b - pad):min(len(y), e + pad)]
    if len(y) < int(0.2 * SR) or np.abs(y).max() < 1e-4:
        raise RuntimeError(f"{src}: empty or near-silent output ({len(y) / SR:.2f} s)")
    m = pyln.Meter(SR, block_size=min(0.4, len(y) / SR * 0.9))
    y = pyln.normalize.loudness(y, m.integrated_loudness(y), -20.0)
    pk = np.abs(y).max()
    if pk > 0.891:
        y = y * (0.891 / pk)
    sf.write(dst, y.astype(np.float32), SR, subtype="PCM_16")


def f0_stats(path):
    import soundfile as sf, librosa
    y, _ = sf.read(path, dtype="float32")
    f0, vf, _ = librosa.pyin(y, fmin=50, fmax=600, sr=SR, frame_length=1200, hop_length=240)
    f = f0[vf & np.isfinite(f0)]
    if len(f) < 10:
        return dict(f0_std_st=float("nan"), f0_range_st=float("nan"), f0_median_hz=float("nan"))
    st = 12 * np.log2(f / np.median(f))
    return dict(f0_std_st=float(st.std()), f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
                f0_median_hz=float(np.median(f)))


def dnsmos(sess, path):
    import soundfile as sf, librosa
    y, _ = sf.read(path, dtype="float32")
    y = librosa.resample(y, orig_sr=SR, target_sr=16000, res_type="soxr_hq")
    L = int(9.01 * 16000)
    while len(y) < L:
        y = np.append(y, y)
    hops = int(np.floor(len(y) / 16000) - 9.01) + 1
    sig, bak, ovr = [], [], []
    for i in range(max(1, hops)):
        seg = y[i * 16000:i * 16000 + L]
        if len(seg) < L:
            continue
        s, b, o = sess.run(None, {"input_1": seg[None].astype(np.float32)})[0][0]
        sig.append(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(s))
        bak.append(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(b))
        ovr.append(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(o))
    return dict(dnsmos_sig=float(np.mean(sig)), dnsmos_bak=float(np.mean(bak)), dnsmos_ovrl=float(np.mean(ovr)))


_NZ = None
FILLERS = {"hm", "hmm", "hem", "hum", "mm", "mhm", "um", "uh", "er", "ah"}


def norm_text(x):
    """Whisper's EnglishTextNormalizer, then two benchmark rules applied to BOTH sides: a '.' or ':' between digits
    becomes a space (Whisper writes the spoken "three thirty" as "3.30" or "3:30"), and interjection spellings
    (hm / hmm / hem / hum ...) are dropped, as the normalizer already drops some of them."""
    global _NZ
    import re
    if _NZ is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _NZ = EnglishTextNormalizer()
    x = re.sub(r"(?<=\d)[.:](?=\d)", " ", _NZ(x))
    return " ".join(w for w in x.split() if w not in FILLERS)


def wer_fields(pid, hyp):
    r, h = norm_text(TEXT[pid]), norm_text(hyp)
    we, wn = edits(r, h); ce, cn = char_edits(r, h)
    return dict(ref_norm=r, hyp_norm=h, w_err=we, w_n=wn, c_err=ce, c_n=cn)


def edits(ref, hyp):
    import jiwer
    if not ref:
        return 0, 0
    o = jiwer.process_words(ref, hyp if hyp else "<empty>")
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def char_edits(ref, hyp):
    import jiwer
    r, h = ref.replace(" ", ""), (hyp or "").replace(" ", "")
    if not h:
        return len(r), len(r)
    o = jiwer.process_characters(r, h)
    return o.substitutions + o.deletions + o.insertions, len(r)


if not a.report_only:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    # 1) normalise + showcase copies
    show = {p["id"] for p in P if p.get("showcase")}
    for s in systems:
        od = os.path.join(norm_dir, s); os.makedirs(od, exist_ok=True)
        for pid in PID:
            src, dst = os.path.join(a.raw, s, pid + ".wav"), os.path.join(od, pid + ".wav")
            if not os.path.exists(src):
                print("MISSING", s, pid); continue
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                try:
                    normalise(src, dst)
                except RuntimeError as e:
                    print("SKIP", e); continue
                PER.setdefault(s, {}).pop(pid, None)       # re-score changed audio
            if pid in show:
                ad = os.path.join(HERE, "audio", s); os.makedirs(ad, exist_ok=True)
                shutil.copy(dst, os.path.join(ad, pid + ".wav"))
    todo = lambda key: [(s, pid) for s in systems for pid in PID
                        if os.path.exists(os.path.join(norm_dir, s, pid + ".wav")) and key not in PER.get(s, {}).get(pid, {})]
    put = lambda s, pid, d: PER.setdefault(s, {}).setdefault(pid, {}).update(d)
    # 2) F0 (CPU workers)
    jobs = todo("f0_std_st")
    if jobs:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
            for (s, pid), r in zip(jobs, ex.map(f0_stats, [os.path.join(norm_dir, s, pid + ".wav") for s, pid in jobs], chunksize=4)):
                put(s, pid, r)
        save_per(); print("F0 done", len(jobs), flush=True)
    # 3) DNSMOS
    jobs = todo("dnsmos_ovrl")
    if jobs and os.path.exists(a.dnsmos):
        import onnxruntime as ort
        sess = ort.InferenceSession(a.dnsmos, providers=["CPUExecutionProvider"])
        for s, pid in jobs:
            put(s, pid, dnsmos(sess, os.path.join(norm_dir, s, pid + ".wav")))
        save_per(); print("DNSMOS done", len(jobs), flush=True)
    # 4) ASR
    jobs = todo("asr")
    if jobs:
        import torch, soundfile as sf, librosa
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        name = a.asr if "/" in a.asr else f"openai/whisper-{a.asr}"
        dev = DEV or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if dev in ("cuda", "mps") else torch.float32
        proc = WhisperProcessor.from_pretrained(name)
        wm = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dt).to(dev).eval()
        for s, pid in jobs:
            y, _ = sf.read(os.path.join(norm_dir, s, pid + ".wav"), dtype="float32")
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000, res_type="soxr_hq")
            feats = proc(y16, sampling_rate=16000, return_tensors="pt").input_features.to(dev).to(dt)
            with torch.no_grad():
                ids = wm.generate(feats, language="en", task="transcribe", num_beams=5, do_sample=False, max_new_tokens=200)
            put(s, pid, dict(asr=proc.batch_decode(ids, skip_special_tokens=True)[0].strip()))
        del wm; save_per(); print("ASR done", len(jobs), flush=True)
    # 5) UTMOSv2
    jobs = todo("utmosv2")
    if jobs:
        import torch, utmosv2
        dev = DEV or ("cuda:0" if torch.cuda.is_available() else "cpu")
        um = utmosv2.create_model(pretrained=True, device=dev)
        # UTMOSv2 crops/samples randomly on every call (per-clip SD about 0.25, about 0.03 on a 54-clip mean): seed it so a
        # re-run reproduces; BENCH_UTMOSV2_REPS>1 averages that many draws per clip (the published rows used 1)
        import random; random.seed(0); np.random.seed(0); torch.manual_seed(0)
        reps = int(os.environ.get("BENCH_UTMOSV2_REPS", "1"))
        for s, pid in jobs:
            v = um.predict(input_path=os.path.join(norm_dir, s, pid + ".wav"), num_workers=0, verbose=False, device=dev, num_repetitions=reps)
            put(s, pid, dict(utmosv2=float(np.asarray(v).reshape(-1)[0])))
        del um; save_per(); print("UTMOSv2 done", len(jobs), flush=True)
    # 6) UTMOS22 strong
    jobs = [] if a.skip_utmos22 else todo("utmos22")
    if jobs:
        import torch, soundfile as sf, librosa
        u22 = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        for s, pid in jobs:
            y, _ = sf.read(os.path.join(norm_dir, s, pid + ".wav"), dtype="float32")
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000, res_type="soxr_hq")
            with torch.no_grad():
                put(s, pid, dict(utmos22=float(u22(torch.from_numpy(y16)[None].to(next(u22.parameters()).device), 16000).item())))
        save_per(); print("UTMOS22 done", len(jobs), flush=True)
    for s in systems:
        for pid in PID:
            f = os.path.join(norm_dir, s, pid + ".wav")
            if os.path.exists(f):
                import soundfile as sf
                PER[s][pid]["dur_s"] = round(sf.info(f).duration, 3)
    save_per()

# ---------------------------------------------------------------- statistics + report
for s_ in PER:
    for pid, v in PER[s_].items():
        if "asr" in v:
            v.update(wer_fields(pid, v["asr"]))
save_per()
B = 10000
rng = np.random.default_rng(0)
IDX = rng.integers(0, len(PID), size=(B, len(PID)))
MEAN_KEYS = ["utmosv2", "utmos22", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "f0_std_st", "f0_range_st", "dur_s"]


def arr(s, k):
    return np.array([PER[s].get(pid, {}).get(k, np.nan) for pid in PID], dtype=float)


def ci_mean(x):
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return None
    xb = np.where(ok, x, np.nan)[IDX]
    bm = np.nanmean(xb, 1)
    return dict(mean=float(np.nanmean(x)), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)), n=int(ok.sum()))


def ci_ratio(e, n):
    if np.nansum(n) == 0:
        return None
    eb, nb = e[IDX].sum(1), n[IDX].sum(1)
    return dict(mean=float(e.sum() / n.sum()), lo=float(np.percentile(eb / nb, 2.5)), hi=float(np.percentile(eb / nb, 97.5)))


systems_meta = json.load(open(os.path.join(HERE, "systems.json")))
present = [s for s in systems if s in PER]
R = {"prompts": len(PID), "post_processing": "24 kHz, trim top_db 40 + 50 ms pad, -20 LUFS, peak -1 dBFS",
     "asr": f"openai/whisper-{a.asr} (transformers, beam 5)", "bootstrap": f"{B} prompt resamples, percentile 95% CI", "systems": {}}
ref = a.ref_system if a.ref_system in present else None
for s in present:
    d = {k: ci_mean(arr(s, k)) for k in MEAN_KEYS}
    d["wer"] = ci_ratio(np.nan_to_num(arr(s, "w_err")), np.nan_to_num(arr(s, "w_n")))
    d["cer"] = ci_ratio(np.nan_to_num(arr(s, "c_err")), np.nan_to_num(arr(s, "c_n")))
    if ref and s != ref:
        dd = {}
        for k in ("utmosv2", "utmos22", "dnsmos_ovrl", "f0_std_st"):
            x = arr(s, k) - arr(ref, k)
            if np.isfinite(x).any():
                dd[k] = ci_mean(x)
        e1, n1 = np.nan_to_num(arr(s, "w_err")), np.nan_to_num(arr(s, "w_n"))
        e0 = np.nan_to_num(arr(ref, "w_err"))
        if n1.sum():
            db = e1[IDX].sum(1) / n1[IDX].sum(1) - e0[IDX].sum(1) / n1[IDX].sum(1)
            dd["wer"] = dict(mean=float((e1.sum() - e0.sum()) / n1.sum()), lo=float(np.percentile(db, 2.5)), hi=float(np.percentile(db, 97.5)))
        d["delta_vs_" + ref] = dd
    d["meta"] = systems_meta.get(s, {})
    rawmeta = os.path.join(a.raw, s, "meta.json")
    if os.path.exists(rawmeta):
        d["render"] = json.load(open(rawmeta))
    R["systems"][s] = d
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)


def fmt(c, p=2, pct=False):
    if not c:
        return "–"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.{p}f}")
    return f"{f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}]"


order = sorted(present, key=lambda s: (systems_meta.get(s, {}).get("order", 99), s))
L = [f"# Ito bench v1: small and edge English TTS", "",
     f"{len(PID)} prompts (`prompts.json`), every system scored on identical post-processed audio. "
     "Mean [95% bootstrap CI over prompts]. WER/CER in %, corpus-level, Whisper large-v3, normalised text. "
     "F0 SD = voiced pitch standard deviation in semitones (prosody-variability proxy; more is not automatically better).", "",
     "| system | params | MCU | UTMOSv2 | UTMOS22 | DNSMOS OVRL | WER % | CER % | F0 SD (st) | TTFA / RTF on MCU |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for s in order:
    d, m = R["systems"][s], systems_meta.get(s, {})
    L.append(f"| {m.get('name', s)} | {m.get('params_str', '?')} | {m.get('mcu', '?')} | {fmt(d['utmosv2'])} | {fmt(d['utmos22'])} | "
             f"{fmt(d['dnsmos_ovrl'])} | {fmt(d['wer'], pct=True)} | {fmt(d['cer'], pct=True)} | {fmt(d['f0_std_st'])} | {m.get('mcu_speed', '–')} |")
if ref:
    L += ["", f"Paired difference vs {systems_meta.get(ref, {}).get('name', ref)} (system minus Ito chip-exact with the fixed frontend, same prompts; a CI that excludes 0 is a real difference):", "",
          "| system | Δ UTMOSv2 | Δ UTMOS22 | Δ DNSMOS OVRL | Δ WER (pts) | Δ F0 SD |", "|---|---|---|---|---|---|"]
    for s in order:
        if s == ref:
            continue
        dd = R["systems"][s].get("delta_vs_" + ref, {})
        L.append(f"| {systems_meta.get(s, {}).get('name', s)} | {fmt(dd.get('utmosv2'))} | {fmt(dd.get('utmos22'))} | "
                 f"{fmt(dd.get('dnsmos_ovrl'))} | {fmt(dd.get('wer'), pct=True)} | {fmt(dd.get('f0_std_st'))} |")
L += ["", "| system | MACs per s of audio | weights | license | notes |", "|---|---|---|---|---|"]
for s in order:
    m = systems_meta.get(s, {})
    L.append(f"| {m.get('name', s)} | {m.get('macs_per_s', '–')} | {m.get('size', '–')} | {m.get('license', '–')} | {m.get('notes', '')} |")
L += ["", "Mean clip duration (s) / DNSMOS SIG / BAK / F0 range p5–p95 (st):", "",
      "| system | dur | SIG | BAK | F0 range |", "|---|---|---|---|---|"]
for s in order:
    d = R["systems"][s]
    L.append(f"| {systems_meta.get(s, {}).get('name', s)} | {fmt(d['dur_s'])} | {fmt(d['dnsmos_sig'])} | {fmt(d['dnsmos_bak'])} | {fmt(d['f0_range_st'])} |")
SC = []
for s in order:
    d, m = R["systems"][s], systems_meta.get(s, {})
    g = lambda k: (d.get(k) or {}).get("mean")
    SC.append(dict(system=s, name=m.get("name", s).replace("*", ""), params=m.get("params"), mcu=m.get("mcu_class", "no"),
                   utmosv2=g("utmosv2"), utmosv2_ci=[(d.get("utmosv2") or {}).get("lo"), (d.get("utmosv2") or {}).get("hi")],
                   utmos22=g("utmos22"), dnsmos_ovrl=g("dnsmos_ovrl"), wer=g("wer"), f0_std_st=g("f0_std_st")))
json.dump(dict(x="params (log scale; rule-based systems have params null)", y="utmosv2", points=SC),
          open(os.path.join(HERE, "scatter.json"), "w"), indent=1)
open(os.path.join(HERE, "work", "results_table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
