# Global and local transformer are adjusted from https://github.com/shengfly/global-local-transformer
import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer, batch normalization, and ReLU activation.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class FeedForward(nn.Module):
    """
    FeedForward network with two convolutional blocks.
    """
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(dim, out_dim, kernel=1, padding=0),
            ConvBlock(out_dim, out_dim, kernel=1, padding=0))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MHSA(nn.Module):
    """Multi-Head Self Attention."""

    def __init__(self, dim: int, head_dim: int = 8,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % head_dim == 0, f"dim({dim}) must be divisible by head_dim({head_dim})"

        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization. """
        nn.init.kaiming_uniform_(self.qkv.weight, nonlinearity='relu')
        nn.init.zeros_(self.qkv.bias)
        nn.init.kaiming_uniform_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        # Apply attention to values and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SMHCA(nn.Module):
    """Simple Multi-Head Cross Attention"""

    def __init__(self, dim: int, out_dim: int, head_dim: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % head_dim == 0, f"dim({dim}) must be divisible by head_dim({head_dim})"

        self.dim = dim
        self.out_dim = out_dim if out_dim else dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # QKV projection
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization. """
        nn.init.kaiming_uniform_(self.q_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.q_proj.bias)
        nn.init.kaiming_uniform_(self.kv_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.kaiming_uniform_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_a: Tensor, x_b: Tensor) -> Tensor:
        B, N, _ = x_a.shape
        M = x_b.size(1)

        # Project queries from x_a
        q = self.q_proj(x_a).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # Project keys/values from x_b
        kv = self.kv_proj(x_b).reshape(B, M, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Compute cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        # Combine and project
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class Instance_bag_transformer(nn.Module):
    """
    Instance-bag transformer with multiple attention blocks.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128,
                 num_blocks: int = 2, drop_rate: float = 0.0):
        super(Instance_bag_transformer, self).__init__()

        self.num_blocks = num_blocks

        # Input projections
        self.global_proj = FeedForward(in_dim * 4, hidden_dim)
        self.local_proj = FeedForward(in_dim, hidden_dim)

        # Transformer components
        self.self_attn_layers = nn.ModuleList([
            MHSA(hidden_dim, head_dim=8, attn_drop=drop_rate, proj_drop=drop_rate)
            for _ in range(num_blocks)
        ])
        self.cross_attn_layers = nn.ModuleList([
            SMHCA(hidden_dim, out_dim=hidden_dim, head_dim=8, attn_drop=drop_rate, proj_drop=drop_rate)
            for _ in range(num_blocks)
        ])
        self.ffn_layers = nn.ModuleList([
            FeedForward(hidden_dim * 3, hidden_dim)
            for _ in range(num_blocks)
        ])

    def _to_sequence(self, x: Tensor) -> Tensor:
        """Convert 2D spatial features to 1D sequence format."""
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)

    def _to_spatial(self, x: Tensor, shape: torch.Size) -> Tensor:
        """Convert 1D sequence back to 2D spatial format."""
        B, C, H, W = shape
        return x.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, local_feats: list, global_feat: Tensor) -> list:
        # Process global feature
        global_feat = self.global_proj(global_feat)
        global_seq = self._to_sequence(global_feat)

        processed_feats = []

        for feat in local_feats:
            x = self.local_proj(feat)
            orig_shape = x.shape

            for block_idx in range(self.num_blocks):
                # Convert to sequence format
                x_seq = self._to_sequence(x)

                # Self-attention
                self_attn = self.self_attn_layers[block_idx](x_seq)
                self_attn = self._to_spatial(self_attn, orig_shape)

                # Cross-attention
                cross_attn = self.cross_attn_layers[block_idx](x_seq, global_seq)
                cross_attn = self._to_spatial(cross_attn, orig_shape)

                # Feature fusion
                combined = torch.cat([x, self_attn, cross_attn], dim=1)
                x = x + self.ffn_layers[block_idx](combined)

            # Preserve original features
            # processed_feats.append(x)
            processed_feats.append(torch.cat([x, feat], dim=1))

        return processed_feats


# if __name__ == '__main__':
#     B = 6
#     C = 512
#     W = 16
#     H = 32
#     nblock = 2
#
#     input_a = torch.rand(B, C, H, W)
#     input_b = torch.rand(B, C*4, H, W)
#
#     mod = Instance_bag_transformer(C)
#     print(mod)
#
#     output = mod([input_a, input_a, input_a, input_a], input_b)
#
#     print('')