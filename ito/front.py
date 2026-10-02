"""Ito acoustic front (inference): phonemes + style -> durations, F0 / voicing / energy and a 100-bin log-mel at 80 fps.

  tokens --> embedding --> conv text encoder --> forward GRU --> duration head
         --> length regulation --> prosody net (log-F0, voicing, energy) --> mel head (log-mel)

The GRU runs forward only and everything at frame rate is a convolution with a bounded right context, so the chain
streams exactly (see ito.synth)."""
import math
import torch
import torch.nn as nn

from .layers import ConvLN, _up

class Front(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.cfg = cfg
        self.hop, self.sr, self.voiced_hz = d["hop"], d["sr"], d["voiced_hz"]
        td, sd, pdim = m["text_dim"], m["style_dim"], m["pros_dim"]
        self.emb = nn.Embedding(m["n_vocab"], td)
        self.enc = nn.ModuleList([ConvLN(td, m["text_kernel"], (m["text_kernel"] - 1) // 2, 0, 0.0)
                                  for _ in range(m["text_layers"])])
        self.style_proj = nn.Linear(m["style_in"], sd)
        self.text_rnn = m.get("text_rnn", 0)
        if self.text_rnn:
            self.rnn_bidir = bool(m.get("text_rnn_bidir", True))
            self.rnn = nn.GRU(td, self.text_rnn, batch_first=True, bidirectional=self.rnn_bidir)
            self.rnn_proj = nn.Linear((2 if self.rnn_bidir else 1) * self.text_rnn, td)
            nn.init.zeros_(self.rnn_proj.weight); nn.init.zeros_(self.rnn_proj.bias)
        self.sent_pos = bool(m.get("sent_pos", False))
        self.dur_layers = nn.ModuleList([ConvLN(td, m["dur_kernel"], (m["dur_kernel"] - 1) // 2, sd, 0.0)
                                         for _ in range(m["dur_layers"])])
        self.dur_out = nn.Conv1d(td, 1, 1)
        nn.init.zeros_(self.dur_out.weight)
        with torch.no_grad():
            self.dur_out.bias.fill_(math.log(4.0))
        n_hf = td + (3 if self.sent_pos else 2)
        self.pros_in = nn.Conv1d(n_hf, pdim, 1)
        self.pros = nn.ModuleList([ConvLN(pdim, m["pros_kernel"], m["pros_rpad"], sd, 0.0)
                                   for _ in range(m["pros_layers"])])
        self.pros_out = nn.Conv1d(pdim, 3, 1)          # normalised log-F0, voicing logit, normalised energy
        self.pros_lc = m["pros_layers"] * (m["pros_kernel"] - 1 - m["pros_rpad"])
        self.pros_la = m["pros_layers"] * m["pros_rpad"]
        # normalisation statistics (stored with the weights): lf0 mean, lf0 std, energy mean, energy std
        self.register_buffer("stats", torch.tensor([5.3, 0.2, 0.0, 1.0]))
        # mel head
        md, nm = m["mel_dim"], m["n_mels"]
        self.mel_in = nn.Conv1d(n_hf + 3, md, 1)
        self.mel_blocks = nn.ModuleList([ConvLN(md, m["mel_kernel"], m["mel_rpad"], sd, 0.0)
                                         for _ in range(m["mel_layers"])])
        self.mel_out = nn.Conv1d(md, nm, 1)
        self.mel_lc = m["mel_layers"] * (m["mel_kernel"] - 1 - m["mel_rpad"])
        self.mel_la = m["mel_layers"] * m["mel_rpad"]
        self.register_buffer("mel_mean", torch.zeros(nm))
        self.register_buffer("mel_std", torch.ones(nm))

    # ---- text side (per token) ----
    def encode(self, tokens, tmask, style):
        """tokens [B, L] long, tmask [B, 1, L] float, style [B, 256] -> h [B, C, L], s [B, sd], logd [B, L]."""
        s = self.style_proj(style)
        h = self.emb(tokens).transpose(1, 2) * tmask
        for l in self.enc:
            h = l(h, mask=tmask)
        if self.text_rnn:
            lens = tmask[:, 0].sum(1).long().cpu()
            pk = nn.utils.rnn.pack_padded_sequence(h.transpose(1, 2), lens, batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=h.size(-1))
            h = (h + self.rnn_proj(o).transpose(1, 2)) * tmask
        x = h
        for l in self.dur_layers:
            x = l(x, s, tmask)
        logd = self.dur_out(x)[:, 0]
        return h, s, logd

    @staticmethod
    def durations_from_log(logd, tmask):
        return (torch.round(torch.exp(logd)).clamp(min=1) * tmask[:, 0]).long()

    def regulate(self, h, dur):
        """h [B, C, L], dur [B, L] frames -> hf [B, C+2(+1), T] (token features + [position-in-token, log-duration]
        (+ position-in-sentence when sent_pos)), fmask [B, 1, T]."""
        B = h.size(0)
        T = int(dur.sum(1).max())
        extra = 3 if self.sent_pos else 2
        out = h.new_zeros(B, h.size(1) + extra, T)
        fmask = h.new_zeros(B, 1, T)
        for b in range(B):
            d = dur[b]
            idx = torch.repeat_interleave(torch.arange(d.numel(), device=h.device), d)
            n = idx.numel()
            start = torch.cumsum(d, 0) - d
            pos = (torch.arange(n, device=h.device) - start[idx]).float() / d[idx].float()
            out[b, :h.size(1), :n] = h[b][:, idx]
            out[b, h.size(1), :n] = pos
            out[b, h.size(1) + 1, :n] = torch.log(d[idx].float()) / 3.0
            if self.sent_pos:
                out[b, h.size(1) + 2, :n] = torch.arange(n, device=h.device).float() / n
            fmask[b, 0, :n] = 1
        return out, fmask

    # ---- frame side ----
    def prosody(self, hf, s, fmask=None):
        x = self.pros_in(hf)
        for l in self.pros:
            x = l(x, s, fmask)
        return x, _up(self.pros_out(x))          # hidden [B, pdim, T], pred [B, 3, T]

    def pred_to_curves(self, pred):
        v = (pred[:, 1] > 0).float()
        lf0n = pred[:, 0] * v
        f0hz = torch.exp(pred[:, 0] * self.stats[1] + self.stats[0]) * v
        return lf0n, v, pred[:, 2], f0hz

    @staticmethod
    def dec_input(hidden, lf0n, v, en):
        return torch.cat([hidden, lf0n[:, None].to(hidden.dtype), v[:, None].to(hidden.dtype),
                          en[:, None].to(hidden.dtype)], 1)

    def mel_head(self, hf, s, lf0n, v, en, fmask=None):
        x = self.mel_in(self.dec_input(hf, lf0n, v, en))
        for b in self.mel_blocks:
            x = b(x, s, fmask)
        y = _up(self.mel_out(x))
        return y * self.mel_std[None, :, None] + self.mel_mean[None, :, None]

    # ---- inference ----
    @torch.no_grad()
    def text_side(self, tokens, style, dur=None):
        tmask = torch.ones(1, 1, tokens.size(1), device=tokens.device)
        h, s, logd = self.encode(tokens, tmask, style)
        if dur is None:
            dur = self.durations_from_log(logd, tmask)
        hf, _ = self.regulate(h, dur)
        return hf, s, dur
