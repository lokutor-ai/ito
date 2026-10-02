"""Building blocks shared by the acoustic front and the vocoder.

Everything that runs at frame rate is a convolution with a bounded right context, so the whole chain streams exactly
(see ito.synth.Chain.stream_chunks)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(t):
    """float32 for bf16/fp16/fp32 tensors, but keep float64 (the exact streaming checks run in float64)."""
    return t if t.dtype == torch.float64 else t.float()


class ConvLN(nn.Module):
    """Residual: conv (lpad, rpad) -> LayerNorm -> optional FiLM(style) -> GELU -> dropout."""

    def __init__(self, dim, k, rpad, sdim=0, p=0.0):
        super().__init__()
        self.lpad, self.rpad = k - 1 - rpad, rpad
        self.conv = nn.Conv1d(dim, dim, k)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(sdim, 2 * dim) if sdim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.drop = nn.Dropout(p)

    def forward(self, x, s=None, mask=None):
        y = self.conv(F.pad(x, (self.lpad, self.rpad)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        if self.film is not None:
            g, b = self.film(s).unsqueeze(-1).chunk(2, 1)
            y = y * (1 + g) + b
        x = x + self.drop(F.gelu(y))
        return x * mask if mask is not None else x


class ConvNeXtBlock(nn.Module):
    """Vocos-style ConvNeXt block (depthwise conv, LayerNorm, pointwise MLP, layer scale) with a bounded right context."""

    def __init__(self, dim, inter, kernel=7, rpad=1, layer_scale=None):
        super().__init__()
        self.lpad, self.rpad = kernel - 1 - rpad, rpad
        self.dw = nn.Conv1d(dim, dim, kernel, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, inter)
        self.pw2 = nn.Linear(inter, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale if layer_scale else 1e-6))

    def forward(self, x):
        r = x
        x = self.dw(F.pad(x, (self.lpad, self.rpad))).transpose(1, 2)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))) * self.gamma
        return r + x.transpose(1, 2)


class HarmonicSource(nn.Module):
    """Sum of n_harm sines at k*F0 (below Nyquist) plus a little noise; noise only when unvoiced. Sample-exact and
    chunkable: F0 is linearly interpolated frame -> sample, the phase accumulator is carried between chunks, and the
    noise comes from an external buffer."""

    def __init__(self, sr, hop, n_fft, n_harm=8, amp=0.1, noise=0.003):
        super().__init__()
        self.sr, self.hop, self.n_fft, self.n_harm, self.amp, self.noise = sr, hop, n_fft, n_harm, amp, noise
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("k", torch.arange(1, n_harm + 1).double()[None, :, None], persistent=False)

    def interp(self, f0, n0, n1, T):
        """f0 [B, T_avail] (Hz) -> float64 [B, n1-n0] at output samples n0..n1-1 of a T-frame utterance."""
        n = torch.arange(n0, n1, device=f0.device, dtype=torch.float64)
        pos = ((n + 0.5) / self.hop - 0.5).clamp(0, T - 1)
        i0 = pos.floor().long()
        i1 = (i0 + 1).clamp(max=T - 1)
        w = pos - i0.double()
        assert int(i1.max()) < f0.size(1), "source needs F0 beyond what is available (lookahead bug)"
        f = f0.double()
        return f[:, i0] * (1 - w) + f[:, i1] * w

    def signal(self, f0, n0, n1, T, phase0, noise):
        """-> (samples [B, n1-n0] float32, phase at the end [B] float64). phase0 [B] (cycles), noise [B, n1-n0]."""
        f = self.interp(f0, n0, n1, T)
        uv = (f > 1.0).float()
        ph = phase0.double()[:, None] + torch.cumsum(f / self.sr, dim=-1)
        ph_end = ph[:, -1]
        ph_end = ph_end - ph_end.floor()      # wrapped double accumulator (as the C engine)
        ph = (ph - ph.floor())[:, None, :]
        keep = (self.k * f[:, None, :] < self.sr / 2).double()
        h = _up((torch.sin(2 * math.pi * self.k * ph) * keep).sum(1).to(noise.dtype)) / self.n_harm
        return self.amp * h * uv + self.noise * noise * uv + (self.amp / 3) * noise * (1 - uv), ph_end

    def feats_full(self, s, T):
        """s [B, T*hop] -> STFT real+imag [B, n_fft+2, T], analysed in the vocoder's iSTFT time base."""
        s = F.pad(s, (self.hop // 2, self.hop // 2))
        S = torch.stft(s, self.n_fft, self.hop, self.n_fft, self.window, center=True, return_complex=True)[..., :T]
        return torch.cat([S.real, S.imag], 1)

    def feats_range(self, s_known, complete, a, b):
        """Same as feats_full()[..., a:b] but from the source samples generated so far (s_known). Frame t needs
        samples < t*hop + n_fft/2 - hop/2; the right reflect-pad of the full version is never reached for t < T."""
        B = s_known.size(0)
        z = s_known.new_zeros(B, self.hop // 2)
        zs = torch.cat([z, s_known] + ([z] if complete else []), 1)
        p = torch.cat([zs[:, 1:self.n_fft // 2 + 1].flip(1), zs], 1)
        need = (b - 1) * self.hop + self.n_fft
        assert p.size(1) >= need, "harmonic STFT needs source samples beyond what is available"
        S = torch.stft(p[:, a * self.hop: need], self.n_fft, self.hop, self.n_fft, self.window, center=False,
                       return_complex=True)
        return torch.cat([S.real, S.imag], 1)
