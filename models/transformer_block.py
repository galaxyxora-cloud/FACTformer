import torch
import torch.nn as nn
from models.freq_processing import FreqGuidedDeformAttn
from models.feed_forward import EnhancedFeedForward
import torch.nn.functional as F



class TransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 num_points=8,
                 attn_drop=0.05,
                 proj_drop=0.1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = FreqGuidedDeformAttn(
            dim,
            num_heads=num_heads,
            num_points=num_points,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.norm2 = norm_layer(dim)
        self.ffn = EnhancedFeedForward(dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 num_points=8,
                 attn_drop=0.005,
                 proj_drop=0.1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_points=num_points,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

