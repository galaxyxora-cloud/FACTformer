import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedFeedForward(nn.Module):
    def __init__(self, dim: int, expand_ratio: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)


        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)

        # 3x3 DWConv: 必不可少，保证 FFN 知道"像素邻居"的信息
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3,
                                padding=1, groups=hidden_dim * 2, bias=bias)

        self.attn_dim = hidden_dim //4

        self.gate_reducer = nn.Conv2d(hidden_dim, self.attn_dim, kernel_size=1, bias=bias)
        self.gate_expander = nn.Conv2d(self.attn_dim, hidden_dim, kernel_size=1, bias=bias)

        self.axial_norm = nn.LayerNorm(self.attn_dim)
        self.qkv_proj = nn.Linear(self.attn_dim, self.attn_dim * 3, bias=bias)
        self.axial_temp = nn.Parameter(torch.ones(1) * 0.5)

        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def _axial_global_look(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        x_seq = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 32]
        x_norm = self.axial_norm(x_seq)

        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        t = torch.clamp(F.softplus(self.axial_temp), min=0.1, max=5.0)
        q = q / t
        q_row = q.view(B * H, W, C)
        k_row = k.view(B * H, W, C)
        v_row = v.view(B * H, W, C)
        row_out = F.scaled_dot_product_attention(q_row, k_row, v_row).view(B, H, W, C)
        q_col = q.permute(0, 2, 1, 3).contiguous().reshape(B * W, H, C)
        k_col = k.permute(0, 2, 1, 3).contiguous().reshape(B * W, H, C)
        v_col = v.permute(0, 2, 1, 3).contiguous().reshape(B * W, H, C)
        col_out = F.scaled_dot_product_attention(q_col, k_col, v_col)
        col_out = col_out.view(B, W, H, C).permute(0, 2, 1, 3)

        return (row_out + col_out).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.project_in(x)
        x = self.dwconv(x)

        x_content, x_gate = x.chunk(2, dim=1)


        gate_global = self.gate_reducer(x_gate)
        gate_global = self._axial_global_look(gate_global)
        gate_global = self.gate_expander(gate_global)

        final_gate = torch.sigmoid(x_gate + gate_global)

        # (Gated Mixing)
        x_fused = F.gelu(x_content) * final_gate
        out = self.project_out(x_fused)

        return out