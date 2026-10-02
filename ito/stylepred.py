"""Optional text -> style predictor (0.2 M parameters).

Ito's default is one FIXED style vector (what the chip uses). This small MLP instead predicts a style from the front's
own style-free text encoding, pooled over the WHOLE sentence. The chip firmware does not use it, because waiting for
the whole sentence would make the time to first audio grow with sentence length; its measured effect was small
(-0.009 log-mel distance to the reference). The blind-test clips of Ito (results/blind9) were rendered WITH it."""
import math
import torch
import torch.nn as nn


class StylePred(nn.Module):
    def __init__(self, d_in=128, d_h=256, d_out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_in + 1, d_h), nn.GELU(), nn.Linear(d_h, d_h), nn.GELU(),
                                 nn.Linear(d_h, d_out))
        self.register_buffer("mu", torch.zeros(d_out))

    def forward(self, h):          # h [B, C, L] (one utterance per row, no padding)
        L = torch.full((h.size(0), 1), math.log(h.size(-1)) / 5.0, device=h.device)
        return self.mu + self.net(torch.cat([h.mean(-1), h.amax(-1), L], 1))


def text_h(front, tok):
    """The front's style-free text encoding (encoder + GRU + projection) of one utterance."""
    tmask = torch.ones(1, 1, tok.size(1), device=tok.device)
    s0 = torch.zeros(1, front.style_proj.in_features, device=tok.device)
    h, _, _ = front.encode(tok, tmask, s0)
    return h
