import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import math

from models.modules.norm import ConvRMSNorm, RMSNorm
from models.modules.conformer import ConformerConvBlock

def exists(x):
    return x is not None

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        time_cond_dim = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        inner_dim = int(dim * mult * 2 / 3)
        dim_out = dim

        self.time_cond = None

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim*2),
            GEGLU(),
            nn.Linear(inner_dim, dim_out)
        ) 

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )

            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.net[-1])


    def forward(self, x, time = None):
        x = self.norm(x)
        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift

        x = self.net(x)

        if exists(self.time_cond):
            x = x*gate

        return x

    
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.emb.weight, std=.01)

    def forward(self, x, pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb
    
# From https://github.com/lucidrains/x-transformers/blob/c7cc22268c8ebceef55fe78343197f0af62edf18/x_transformers/x_transformers.py#L272
class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth=2, log_distance = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
    
class Attention(nn.Module):
    def __init__(
            self, 
            dim, 
            dim_head = 32,
            time_cond_dim = None,
            dropout=0.
            ):
        super().__init__()
        assert dim % dim_head == 0, 'Dimension must be divisible by the head dimension'
        self.heads = dim // dim_head

        self.dropout = dropout
        self.time_cond = None

        self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = self.heads, log_distance = False, depth = 2)

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.to_out)

    def forward(self, x, time=None, audio_mask=None):
        b, c, n = x.shape

        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift
        
        qkv = self.to_qkv(x).chunk(3, dim = 2)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads).contiguous(), qkv)

        i, j = map(lambda t: t.shape[-2], (q, k))

        attn_bias = self.rel_pos(i, j)
        attn_bias = repeat(attn_bias, 'h i j -> b h i j', b=b)

        if exists(audio_mask):
            mask_value = -torch.finfo(q.dtype).max
            mask = rearrange(audio_mask, 'b l -> b () () l')
            attn_bias = attn_bias.masked_fill(~mask, mask_value)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0., attn_mask=attn_bias)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        if exists(self.time_cond):
            out = out*gate

        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, dim_context, dim_head = 32, time_cond_dim=None, dropout=0.):
        super().__init__()
        assert dim % dim_head == 0, 'Dimension must be divisible by the head dimension'
        self.heads = dim // dim_head
        self.dropout = dropout
        self.norm = RMSNorm(dim)
        self.time_cond = None
        self.time_cond = None
        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        self.time_cond = None        
        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])


        self.norm_context = nn.LayerNorm(dim_context)


        self.null_kv = nn.Parameter(torch.randn(2, dim))
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 3),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[-2])
        else:
            zero_init_(self.to_out)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

    def forward(self, x, context, context_mask, time=None):
        '''
        x: [B, L_audio, d_unet]
        context: [B, L_text, d_lm]
        context_mask: [B, L_text]
        '''
        b, c, n = x.shape
        x = self.norm(x)
        if exists(self.time_cond):
            assert exists(time)
            scale, shift, gate = self.time_cond(time).chunk(3, dim = 2)
            x = (x * (scale + 1)) + shift
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads).contiguous(), (q, k, v))
        # Null value for classifier free guidance
        
        nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h=self.heads), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        query_len = q.shape[2]

        # RMSNorm Trick for stability
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Masking pad tokens 
        context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = repeat(context_mask, 'b j -> b h q_len j', h=self.heads, q_len=query_len)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=context_mask, dropout_p=self.dropout if self.training else 0.)
        # attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        # attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
        # out = attn_weight @ v

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        if exists(self.time_cond):
            out = out*gate

        return out


class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        *,
        num_layers,
        time_cond_dim,
        dim_head = 64,
        ff_mult = 4,
        dropout=0.0,
        conformer=False,
    ):
        super().__init__()
        self.dim = dim

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, dropout=dropout),
                CrossAttention(dim = dim, dim_head = dim_head, dim_context=dim_context, dropout=dropout),
                ConformerConvBlock(dim, time_cond_dim=time_cond_dim, channels_first=False) if conformer else None,
                FeedForward(dim=dim, mult=ff_mult, time_cond_dim=time_cond_dim)
            ]))


    def forward(
        self,
        x,
        *,
        time,
        context,
        context_mask,
        audio_mask,
    ):
        for attn, cross_attn, conv, ff in self.layers:
            res = x
            x = attn(x, audio_mask=audio_mask) + res

            res = x
            x = cross_attn(x, context = context,  context_mask=context_mask) + res

            if conv is not None:
                res = x
                x = conv(x, time=time) + res

            res = x
            x = ff(x, time=time) + res

        return x
