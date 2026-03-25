import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashAttention(nn.Module):
    """
    Block-wise scaled dot-product attention.
    Each query block attends to ALL key/value blocks (global attention,
    computed in tiles for memory efficiency), matching paper Eq. 11.
    block_size=5 per Table II.
    """
    def __init__(self, embed_dim, num_heads=1, block_size=5):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.block_size = block_size
        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, E = x.shape
        Q = self.q_proj(x)   # (B,T,E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scale   = math.sqrt(E)
        outputs = []
        n_blocks = (T + self.block_size - 1) // self.block_size  # handles non-divisible T

        for i in range(n_blocks):
            q_block = Q[:, i*self.block_size:(i+1)*self.block_size, :]  # (B,bs,E)

            # ── attend to ALL keys (global, computed block by block for memory) ──
            scores_list = []
            for j in range(n_blocks):
                k_block = K[:, j*self.block_size:(j+1)*self.block_size, :]
                s = torch.matmul(q_block, k_block.transpose(-2, -1)) / scale
                scores_list.append(s)

            # concat along key dim → (B, bs, T)
            scores = torch.cat(scores_list, dim=-1)

            # numerically stable softmax (log-sum-exp trick, Eq.11)
            m = scores.max(dim=-1, keepdim=True).values
            scores = scores - m
            attn = torch.softmax(scores, dim=-1)   # (B,bs,T)

            # weighted sum over all values
            out_block = torch.matmul(attn, V)      # (B,bs,E)
            outputs.append(out_block)

        x_out = torch.cat(outputs, dim=1)          # (B,T,E)
        return self.out_proj(x_out)
