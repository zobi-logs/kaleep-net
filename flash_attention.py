import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    """
    Simplified FlashAttention
    """
    def __init__(self, embed_dim, num_heads=1, block_size=16):
        super(FlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: (batch, seq, embed_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Block-wise attention
        B, T, E = Q.size()
        blocks = T // self.block_size
        outputs = []
        for i in range(blocks):
            q_block = Q[:, i*self.block_size:(i+1)*self.block_size, :]
            k_block = K[:, i*self.block_size:(i+1)*self.block_size, :]
            v_block = V[:, i*self.block_size:(i+1)*self.block_size, :]
            attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (E ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_block)
            outputs.append(attn_out)
        x_out = torch.cat(outputs, dim=1)
        return self.out_proj(x_out)
