from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, LayerNorm
from einops.layers.torch import Rearrange, Reduce


def pair(x): return x if isinstance(x, tuple) else (x, x)


class PreNormResidual(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim, bias=False)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, dim_hidden, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim_hidden, dim),
        nn.Dropout(dropout)
    )


def MLPMixer1D(*, dim, depth, seq_len, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(seq_len, int(
                expansion_factor * dim), dropout, chan_first)),
            PreNormResidual(dim, FeedForward(
                dim, int(expansion_factor_token * dim), dropout, chan_last))
        ) for _ in range(depth)],
        LayerNorm(dim, bias=False)
    )


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
# quick test


class Attention(nn.Module):
    """
    Self-Attention cho sequence
    Input : (B, N, D)
    Output: (B, N, D)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (B, N, D)
        mask: (B, N) hoặc (B, 1, 1, N)
        """

        B, N, D = x.shape

        # ---- QKV ----
        qkv = self.to_qkv(x)                       # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)             # each (B, N, D)

        # ---- reshape heads ----
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, H, N, Dh)

        # ---- attention ----
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)

        if mask is not None:
            # mask = True → keep, False → mask out
            if mask.dim() == 2:
                mask = mask[:, None, None, :]          # (B,1,1,N)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ---- aggregate ----
        out = attn @ v                                 # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.to_out(out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Attention + MLP block
    Input : (B, N, D)
    Output: (B, N, D)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=dropout
        )

    def forward(self, x, mask=None):
        # ---- Attention block ----
        x = x + self.attn(self.norm1(x), mask=mask)

        # ---- MLP block ----
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Stack many TransformerBlock
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


if __name__ == '__main__':

    # tokens = torch.randn(1, 1024, 512)
    # mixer = MLPMixer1D(dim=512, depth=4, seq_len=1024)
    # output = mixer(tokens)
    # print('output shape:', output.shape)
    # print(mixer)

    # B = 2
    # img_size = 224
    # patch_size = 16
    # embed_dim = 768

    # x = torch.randn(B, 3, img_size, img_size)

    # patch_embed = PatchEmbed(
    #     img_size=img_size,
    #     patch_size=patch_size,
    #     in_chans=3,
    #     embed_dim=embed_dim
    # )

    # out = patch_embed(x)

    # expected_num_patches = (img_size // patch_size) ** 2

    # print("Input shape :", x.shape)
    # print("Output shape:", out.shape)

    # assert out.shape == (B, expected_num_patches, embed_dim)
    # print("Shape test passed!")

    B, N, D = 2, 16, 64
    num_heads = 8

    x = torch.randn(B, N, D)

    attn = Attention(
        dim=D,
        num_heads=num_heads,
        dropout=0.1
    )

    out = attn(x)

    print("=== Attention Test ===")
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    assert out.shape == x.shape
    print("✅ Attention output shape correct")

    # ---- test Attention with mask ----
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -4:] = False   # mask 4 token cuối

    out_masked = attn(x, mask=mask)

    print("\n=== Attention with Mask ===")
    print("Output shape:", out_masked.shape)

    assert out_masked.shape == x.shape
    print("✅ Attention mask works")


    mlp = Mlp(
    in_features=D,
    hidden_features=D * 4,
    drop=0.1
    )

    out = mlp(x)

    print("\n=== MLP Test ===")
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    assert out.shape == x.shape
    print("✅ MLP output shape correct")