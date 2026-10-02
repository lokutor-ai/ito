"""Ito vocoder: log-mel (24 kHz, hop 300, 100 HTK mel bins, natural log) + F0 -> 24 kHz audio.

A Vocos-style network at 80 fps: input [log-mel (100) | normalised log-F0 * voiced | voiced] -> 7-tap embedding conv
-> LayerNorm -> + projected STFT of an 8-harmonic F0 source -> 5 ConvNeXt blocks (bounded right context) -> LayerNorm
-> linear head to log-magnitude + phase -> iSTFT (n_fft 1200, hop 300, frames aligned on the half hop).
It is a pure vocoder: everything acoustic is in the mel; F0 only drives the harmonic source and two input channels.
Lookahead: embedding rpad + n_blocks * rpad frames of input, + 3 frames for the iSTFT overlap-add."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvNeXtBlock, HarmonicSource, _up

class Vocoder(nn.Module):
    def __init__(self, n_mels=100, dim=256, inter=768, n_blocks=5, kernel=7, rpad=1, n_fft=1200, hop=300, sr=24000,
                 n_harm=8, voiced_hz=60.0):
        super().__init__()
        self.n_fft, self.hop, self.rpad, self.voiced_hz = n_fft, hop, rpad, voiced_hz
        self.embed = nn.Conv1d(n_mels + 2, dim, 7)
        self.norm0 = nn.LayerNorm(dim, eps=1e-6)
        self.src = HarmonicSource(sr, hop, n_fft, n_harm)
        self.harm_proj = nn.Conv1d(n_fft + 2, dim, 1)
        nn.init.zeros_(self.harm_proj.weight); nn.init.zeros_(self.harm_proj.bias)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, inter, kernel, rpad, 1.0 / n_blocks) for _ in range(n_blocks)])
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("f0stats", torch.tensor([5.3, 0.2]))   # log-F0 mean / std (stored with the weights)
        # receptive field in frames (left, right) of spec() with respect to its inputs
        self.lc = (6 - rpad) + n_blocks * (kernel - 1 - rpad)
        self.la = rpad + n_blocks * rpad

    def spec(self, xin, s, hfeat):          # s is unused (no style: pure vocoder); kept for a uniform signature
        x = self.embed(F.pad(xin, (6 - self.rpad, self.rpad)))
        x = self.norm0(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.harm_proj(hfeat.to(x.dtype))
        for b in self.blocks:
            x = b(x)
        return _up(self.out(self.norm1(x.transpose(1, 2))).transpose(1, 2))   # [B, n_fft+2, T]

    def complex_spec(self, x):
        nb = self.n_fft // 2 + 1
        mag = torch.exp(x[:, :nb]).clamp(max=1e2)
        return torch.polar(mag, x[:, nb:])

    def istft(self, x):
        T = x.size(-1)
        y = torch.istft(self.complex_spec(x), self.n_fft, self.hop, self.n_fft, self.window, center=True,
                        length=T * self.hop + self.hop // 2 + self.hop // 4)
        return y[:, self.hop // 2: self.hop // 2 + T * self.hop]   # frame t -> samples [t*hop, (t+1)*hop)

    def clean_f0(self, f0hz):
        return f0hz * (f0hz > self.voiced_hz).float()

    def feats(self, mel, f0hz):
        """mel [B, 100, T] log-mel, f0hz [B, T] (0 = unvoiced) -> vocoder input [B, 102, T]."""
        v = (f0hz > self.voiced_hz).float()
        lf0n = (torch.log(_up(f0hz).clamp(min=1.0)) - self.f0stats[0]) / self.f0stats[1] * v
        return torch.cat([mel, lf0n[:, None].to(mel.dtype), v[:, None].to(mel.dtype)], 1)

    def forward(self, mel, f0hz, phase0=None, noise=None):
        f0hz = self.clean_f0(_up(f0hz))
        xin = self.feats(mel, f0hz)
        B, _, T = xin.shape
        N = T * self.hop
        if phase0 is None:
            phase0 = torch.rand(B, device=xin.device, dtype=torch.float64)
        if noise is None:
            noise = torch.randn(B, N, device=xin.device)
        with torch.autocast("cuda", enabled=False):
            src, _ = self.src.signal(_up(f0hz), 0, N, T, phase0, noise)
            hfeat = self.src.feats_full(src, T)
        x = self.spec(xin, None, hfeat)
        with torch.autocast("cuda", enabled=False):
            return self.istft(x)

