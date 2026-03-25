import torch
import torch.nn as nn

class KANLayer(nn.Module):
    """
    KAN layer matching paper Eqs. 3 & 5.
    h, b: (in_features, n_basis)        — scale/bias per input per basis
    w:    (in_features, n_basis, out_features) — mixing weights
    """
    def __init__(self, in_features, out_features, n_basis=16):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_basis      = n_basis

        self.h = nn.Parameter(torch.randn(in_features, n_basis) * 0.05)
        self.b = nn.Parameter(torch.zeros(in_features, n_basis))
        self.w = nn.Parameter(torch.randn(in_features, n_basis, out_features) * 0.05)

    def forward(self, x):
        # x: (B, T, D)
        x_exp = x.unsqueeze(-1)                          # (B,T,D,1)
        h = self.h.view(1, 1, self.in_features, self.n_basis)
        b = self.b.view(1, 1, self.in_features, self.n_basis)
        basis = torch.tanh(h * x_exp + b)               # (B,T,D,K)
        out   = torch.einsum("btdk,dko->bto", basis, self.w)  # (B,T,O)
        return out
