import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from models.modules.norm import RMSNorm, ConvRMSNorm

def exists(x):
    return x is not None

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class ConformerConvBlock(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        dim,
        dim_out=None,
        depthwise_kernel_size=17,
        expansion_factor=2,
        time_cond_dim = None,
        channels_first=True,
        zero_init=True,
    ):
        """
        Args:
            dim: Embedding dimension
            depthwise_kernel_size: Depthwise conv layer kernel size
        """
        super(ConformerConvBlock, self).__init__()
        dim_out = default(dim_out, dim)
        self.channels_first = channels_first
        inner_dim = dim * expansion_factor

        self.time_cond = None
        self.time_gate = exists(time_cond_dim) and zero_init
        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        

        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, f"kernel_size: {depthwise_kernel_size} should be a odd number for 'SAME' padding"
        self.pointwise_conv1 = torch.nn.Conv1d(
            dim,
            2 * inner_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            inner_dim,
            inner_dim,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=inner_dim
        )
        self.norm = ConvRMSNorm(dim) if self.channels_first else RMSNorm(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = torch.nn.Conv1d(
            inner_dim,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        if (not self.time_gate) and zero_init:
            zero_init_(self.pointwise_conv2)


    def forward(self, x, time = None, scale_shift=None,):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        assert not (exists(self.time_cond) and exists(scale_shift))
        
        x = self.norm(x)
        if exists(self.time_cond):
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift
        elif exists(scale_shift):
            scale, shift, = scale_shift
            x = (x * (scale + 1)) + shift

        if not self.channels_first:
            x = rearrange(x, 'b l c -> b c l')

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*inner_dim, len)
        x = self.glu(x)  # (batch, inner_dim, len)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)

        if not self.channels_first:
            x = rearrange(x, 'b c l -> b l c')
        
        if exists(self.time_cond):
            x = x*gate
        
        return x
