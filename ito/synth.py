"""Ito inference: text -> phonemes -> front (durations, F0 / voicing / energy, log-mel) -> vocoder -> 24 kHz audio.

    from ito import Ito
    tts = Ito.load()                          # fetched from Hugging Face on first use (ito/weights.py)
    tts = Ito.load(voice="g")                 # the male voice (default "d", female)
    wav = tts.synthesize("Hello there!")      # float32 numpy array at tts.sr (24 kHz)
    for chunk in tts.stream("Hello there!"):  # 100 ms chunks, computed with bounded lookahead, as on the chip
        ...

Streaming (Chain.stream_chunks) evaluates each stage on windows with its full left receptive field (on the chip these
are ring buffers), carries the harmonic-source phase and the iSTFT overlap-add state, and asserts that it never reads
features beyond its lookahead. The result equals whole-utterance synthesis (Chain.infer_full) to float precision."""
import math
import os

import numpy as np
import torch
import torch.nn as nn

from .front import Front
from .vocoder import Vocoder



class Chain(nn.Module):
    """Front + vocoder (the PyTorch twin of the chip's graph)."""

    def __init__(self, front, voc):
        super().__init__()
        self.front, self.voc = front, voc
        self.hop = voc.hop
        self._p64 = None

    @torch.no_grad()
    def text_side(self, tokens, style, dur=None):
        return self.front.text_side(tokens, style, dur)

    @torch.no_grad()
    def prosody(self, hf, s):
        """The prosody net in float64. F0 feeds the harmonic source's phase ACCUMULATOR, which integrates any input
        difference: float32 convs evaluated on windows of different lengths (streaming vs whole) differ by ~1e-5 and
        the integrated phase would drift. In float64 the F0 curve is identical in both modes after the cast back."""
        if self._p64 is None:
            import copy
            self._p64 = copy.deepcopy(nn.ModuleDict(dict(pros_in=self.front.pros_in, pros=self.front.pros,
                                                          pros_out=self.front.pros_out))).double()
        p = self._p64
        x = p["pros_in"](hf.double())
        sd = s.double()
        for l in p["pros"]:
            x = l(x, sd)
        return p["pros_out"](x).to(hf.dtype if hf.dtype == torch.float64 else torch.float32)

    @torch.no_grad()
    def frames_full(self, hf, s):
        pred = self.prosody(hf, s)
        lf0n, v, en, f0hz = self.front.pred_to_curves(pred)
        return self.front.mel_head(hf, s, lf0n, v, en), f0hz, pred

    @torch.no_grad()
    def infer_full(self, hf, s, phase0, noise):
        mel, f0hz, pred = self.frames_full(hf, s)
        return self.voc(mel, f0hz, phase0, noise), mel, f0hz

    @torch.no_grad()
    def stream_chunks(self, hf, s, phase0, noise, chunk=8, cut_istft=0, cut_voc=0, cut_mel=0, cut_pros=0, info=None):
        """Yield audio [B, n*hop] for frames [s0, s0+chunk) in order. cut_* remove frames of lookahead from one stage
        (negative controls for tests only: the output must then differ). info['max_la'] = lookahead used (frames)."""
        fr, voc, src = self.front, self.voc, self.voc.src
        voc_la, mel_la, pros_la = voc.la - cut_voc, fr.mel_la - cut_mel, fr.pros_la - cut_pros
        B, _, T = hf.shape
        hop, n_fft = self.hop, voc.n_fft
        pred = hf.new_zeros(B, 3, T); pros_done = 0
        mel = hf.new_zeros(B, fr.mel_out.out_channels, T); mel_done = 0
        s_buf = hf.new_zeros(B, 0); phase = phase0.double(); src_n = 0
        xspec = hf.new_zeros(B, n_fft + 2, T); spec_done = 0
        L = (T + 1) * hop + n_fft
        ybuf = hf.new_zeros(B, L); env = hf.new_zeros(L); ola_done = 0
        w = voc.window; w2 = w * w
        max_la = 0
        for s0 in range(0, T, chunk):
            e = min(T, s0 + chunk)
            need_spec = min(T, e + 3 - cut_istft)                 # iSTFT: output of frame e-1 needs spec <= e+2
            need_x = min(T, need_spec + voc_la)                  # vocoder input frames (mel + F0 channels)
            need_src = min(T * hop, (need_x - 1) * hop + n_fft // 2 - hop // 2)
            pos = min(T - 1, max(0.0, (need_src - 1 + 0.5) / hop - 0.5))
            need_f0 = min(T, int(math.floor(pos)) + 2)
            need_mel = need_x
            need_pros = max(need_x, need_f0, min(T, need_mel + mel_la))
            avail = min(T, need_pros + pros_la)
            max_la = max(max_la, avail - e)
            hf_vis = hf[..., :avail]
            if need_pros > pros_done:                             # 1) prosody net
                a, b = pros_done, need_pros
                lo, hi = max(0, a - fr.pros_lc), min(avail, b + pros_la)
                assert hi == T or hi >= b + pros_la
                pwin = self.prosody(hf_vis[..., lo:hi], s)
                pred[..., a:b] = pwin[..., a - lo:b - lo]; pros_done = b
            lf0n, v, en, f0hz = fr.pred_to_curves(pred[..., :pros_done])
            f0hz = voc.clean_f0(f0hz)
            if need_mel > mel_done:                               # 2) mel head
                a, b = mel_done, need_mel
                lo, hi = max(0, a - fr.mel_lc), min(pros_done, b + mel_la)
                assert hi == T or hi >= b + mel_la
                mw = fr.mel_head(hf_vis[..., lo:hi], s, lf0n[..., lo:hi], v[..., lo:hi], en[..., lo:hi])
                mel[..., a:b] = mw[..., a - lo:b - lo]; mel_done = b
            if need_src > src_n:                                  # 3) harmonic source samples
                seg, phase = src.signal(f0hz, src_n, need_src, T, phase, noise[:, src_n:need_src])
                s_buf = torch.cat([s_buf, seg], 1); src_n = need_src
            if need_spec > spec_done:                             # 4) vocoder spectral frames
                a, b = spec_done, need_spec
                lo, hi = max(0, a - voc.lc), min(need_x, b + voc_la)
                assert hi == T or hi >= b + voc_la
                assert hi <= mel_done and hi <= f0hz.size(-1)
                xin = voc.feats(mel[..., lo:hi], f0hz[..., lo:hi])
                hfe = src.feats_range(s_buf, src_n == T * hop, lo, hi)
                xw = voc.spec(xin, None, hfe)
                xspec[..., a:b] = xw[..., a - lo:b - lo]; spec_done = b
            if spec_done > ola_done:                              # 5) overlap-add
                S = voc.complex_spec(xspec[..., ola_done:spec_done])
                frm = torch.fft.irfft(S, n=n_fft, dim=1) * w[None, :, None]
                for i, t in enumerate(range(ola_done, spec_done)):
                    ybuf[:, t * hop: t * hop + n_fft] += frm[..., i]
                    env[t * hop: t * hop + n_fft] += w2
                ola_done = spec_done
            if info is not None:
                info["max_la"] = max_la
            o0, o1 = s0 * hop + hop // 2 + n_fft // 2, e * hop + hop // 2 + n_fft // 2
            yield ybuf[:, o0:o1] / env[o0:o1]

    @torch.no_grad()
    def infer_stream(self, hf, s, phase0, noise, chunk=8, **cut):
        """Whole utterance through the streaming path. Returns (audio [B, T*hop], lookahead in frames)."""
        info = {}
        y = torch.cat(list(self.stream_chunks(hf, s, phase0, noise, chunk, info=info, **cut)), 1)
        return y, info["max_la"]


class Ito:
    """The released voice: front + vocoder + the fixed style vector the chip uses."""

    def __init__(self, chain, style, meta=None, stylepred=None):
        self.chain = chain.eval()
        self.style = style            # [1, 256], the fixed style (what the chip uses)
        self.stylepred = stylepred    # optional text -> style predictor (ito.stylepred)
        self.meta = meta or {}
        self.sr = chain.front.sr
        self.hop = chain.hop
        self.device = style.device

    @classmethod
    def load(cls, path=None, device="cpu", voice="d"):
        """voice: 'd' (female, LibriTTS-R 4970; default) or 'g' (male, LibriTTS-R 5105). path overrides voice."""
        if path is None:
            from .weights import weights_path, voice_files
            path = weights_path(voice_files(voice)[0])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        ck = torch.load(path, map_location=device, weights_only=True)
        front = Front(ck["front"]["cfg"]); front.load_state_dict(ck["front"]["model"])
        voc = Vocoder(**ck["vocoder"]["vcfg"]); voc.load_state_dict(ck["vocoder"]["model"])
        chain = Chain(front, voc).to(device)
        sp = None
        if "stylepred" in ck:
            from .stylepred import StylePred
            sp = StylePred(**ck["stylepred"]["cfg"]); sp.load_state_dict(ck["stylepred"]["model"]); sp = sp.to(device).eval()
        return cls(chain, ck["style"].float().reshape(1, -1).to(device), ck.get("meta"), sp)

    def tokens(self, text=None, phonemes=None):
        from .text import phonemes_to_ids, text_to_ids
        ids = phonemes_to_ids(phonemes) if phonemes is not None else text_to_ids(text)
        if len(ids) > 400:
            raise ValueError(f"{len(ids)} tokens; split the text into sentences (the chip stops at 400)")
        return torch.LongTensor(ids)[None].to(self.device)

    @torch.no_grad()
    def _prepare(self, tok, seed, duration_scale, style="mean"):
        fr = self.chain.front
        tmask = torch.ones(1, 1, tok.size(1), device=self.device)
        if style == "mean":
            sty = self.style
        elif style == "predicted":
            if self.stylepred is None:
                raise ValueError("this checkpoint has no style predictor")
            from .stylepred import text_h
            sty = self.stylepred(text_h(fr, tok))
        else:
            raise ValueError("style must be 'mean' (the chip's) or 'predicted'")
        h, s, logd = fr.encode(tok, tmask, sty)
        dur = (torch.round(torch.exp(logd) * duration_scale).clamp(min=1) * tmask[:, 0]).long()
        hf, _ = fr.regulate(h, dur)
        T = hf.size(-1)
        g = torch.Generator(device=self.device); g.manual_seed(seed)
        noise = torch.randn(1, T * self.hop, device=self.device, generator=g)
        phase0 = torch.full((1,), 0.25, device=self.device, dtype=torch.float64)
        return hf, s, phase0, noise

    @torch.no_grad()
    def stream(self, text=None, phonemes=None, seed=0, chunk=8, duration_scale=1.0, style="mean"):
        """Yield float32 numpy chunks of chunk*hop samples (8 frames = 100 ms at 24 kHz).
        style: 'mean' (fixed style, as the chip) or 'predicted' (text -> style predictor, as the blind-test clips)."""
        hf, s, p0, nz = self._prepare(self.tokens(text, phonemes), seed, duration_scale, style)
        for y in self.chain.stream_chunks(hf, s, p0, nz, chunk=chunk):
            yield y[0].float().cpu().numpy()

    @torch.no_grad()
    def synthesize(self, text=None, phonemes=None, seed=0, chunk=8, streaming=False, duration_scale=1.0, style="mean"):
        """-> float32 numpy array at self.sr. streaming=True runs the chunked path of stream() (same audio to ~120 dB)."""
        if streaming:
            return np.concatenate(list(self.stream(text, phonemes, seed, chunk, duration_scale, style)))
        hf, s, p0, nz = self._prepare(self.tokens(text, phonemes), seed, duration_scale, style)
        y, _, _ = self.chain.infer_full(hf, s, p0, nz)
        return y[0].float().cpu().numpy()
