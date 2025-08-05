import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) 
    """
    def __init__(self, in_features, out_features, n_basis=16):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        self.w = nn.Parameter(torch.randn(in_features, n_basis, out_features) * 0.05)
        self.h = nn.Parameter(torch.randn(in_features, n_basis, out_features) * 0.05)
        self.b = nn.Parameter(torch.zeros(in_features, n_basis, out_features))

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        batch, seq_len, in_features = x.shape
        # Expand input to (batch, seq_len, in_features, n_basis)
        x_exp = x.unsqueeze(-1).expand(-1, -1, in_features, self.n_basis)
        # h, b: (in_features, n_basis, out_features)
        h = self.h.permute(2, 0, 1)    # (out_features, in_features, n_basis)
        b = self.b.permute(2, 0, 1)
        # Basis expansion: tanh(h * x + b), then weighted sum
        basis = torch.tanh(self.h * x_exp + self.b)
        # Sum over input features and basis functions
        # basis: (batch, seq_len, in_features, n_basis)
        # w: (in_features, n_basis, out_features)
        out = torch.einsum('bsin,ino->bso', basis, self.w)
        return out  # (batch, seq_len, out_features)
