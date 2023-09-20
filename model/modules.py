import torch
from math import ceil
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

"""
The components for the transformer model
"""

class EncoderBlock(nn.Module):
    """
    Params:
        emb_dim: dimension size of the embeddings
        nhead: number of heads
        dropout: the dropout rate
    Args:
        input: the image input (b, h * w, c)
    Returns:
        x: the output of self-attention and MLP (b, h * w, c)
    """
    def __init__(self, emb_dim:int, nhead:int, dropout:float) -> None:
        super(EncoderBlock, self).__init__()

        self.ln1 = LayerNorm(emb_dim)
        self.self_attn = nn.MultiheadAttention(emb_dim, nhead, dropout, False, batch_first=True)
        self.mlp = MLP(emb_dim)

    def forward(self, input):
        x = self.ln1(input)
        x = input + self.self_attn(x, x, x, need_weights=False, average_attn_weights=False)[0]
        x = self.mlp(x)
        return x
    
class MultiAxisTransformerBlock(nn.Module):
    """
    Params:
        nchannels: the hidden size of the image input
        block_size: the kernel size of the image blocks
        nhead: number of heads
        dropout: dropout rate
    Args:
       input: the image input (b, c, h, w)
       context: the image input (b, h * w, c) 
    Returns:
        y: the final hidden state (b, n, p, c)
    """
    def __init__(self, nchannels:int, block_size:int, nhead:int, dropout:float) -> None:
        super(MultiAxisTransformerBlock, self).__init__()

        self.ln1 = LayerNorm(nchannels)
        self.self_attn = MultiAxisAttention(nchannels, nhead, dropout)
        self.ln2 = LayerNorm(nchannels)
        self.cross_attn = MultiQueryAttention(nchannels, nhead, dropout)
        self.mlp = MLP(nchannels)

    def forward(self, input, context):
        x = input + self.self_attn(self.ln1(input))
        x = x + self.cross_attn(self.ln2(x), context)
        x = self.mlp(x)
        return x

class TransformerBlock(nn.Module):
    """
    Params:
        emb_dim: dimension size of the embeddings
        nhead: number of heads
        dropout: the dropout rate
    Args:
        input: the query input for the decoder (b, h * w, d)
        context: the key and value input from the encoder (b, s, d)
    Returns:
        x: the last hidden state of the transformer block (b, h * w, d)
    """
    def __init__(self, emb_dim:int, nhead:int, dropout:float) -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = LayerNorm(emb_dim)
        self.self_attn = MultiQueryAttention(emb_dim, nhead, dropout)
        self.ln2 = LayerNorm(emb_dim)
        self.cross_attn = MultiQueryAttention(emb_dim, nhead, dropout)
        self.mlp = MLP(emb_dim)

    def forward(self, input, context):
        x = input + self.self_attn(self.ln1(input), self.ln1(input))
        if context is not None:
            x = x + self.cross_attn(self.ln2(x), context)
        x = self.mlp(x)
        return x

class MLP(nn.Module):
    """
    Params:
        emb_dim: dimension size of the embeddings
    Args:
        x: output embeddings from cross-attention (b, h * w, d)
    Returns:
        x: final hidden state of the embeddings (b, h * w, d)
    """
    def __init__(self, emb_dim:int) -> None:
        super(MLP, self).__init__()

        # inner dimension - per Shazeer's recommendation on GEGLU
        inner_dim = int(emb_dim * 4 * 2/3)
        # all other components of the MLP
        self.ln1 = LayerNorm(emb_dim)
        self.proj1 = nn.Linear(emb_dim, 2 * inner_dim, bias=False)
        self.act = GEGLU()
        self.ln2 = LayerNorm(inner_dim)
        self.proj2 = nn.Linear(inner_dim, emb_dim, bias=False)

    def forward(self, x):
        x = self.proj1(self.ln1(x))
        x = self.act(x)
        x = self.proj2(self.ln2(x))
        return x

class MultiQueryAttention(nn.Module):
    """
    Params:
        emb_dim: dimension size of the embeddings
        nhead: number of heads
        dropout: the dropout rate
    Args:
        input: the query input (b, ..., d)
        context: if specified, cross-attention with another output. Else it is self-attention (b, ..., d)
    Returns:
        y: the output after cross/self-attention (b, ..., d)
    """
    def __init__(self, emb_dim:int, nhead:int, dropout:float) -> None:
        super(MultiQueryAttention, self).__init__()

        # cache numerical information for the rest of the operation
        self.nhead = nhead
        self.emb_dim = emb_dim
        self.scale = (emb_dim // nhead) ** -0.5

        # projection layers
        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, 2 * emb_dim // nhead, bias=False)
        self.to_o = nn.Linear(emb_dim, emb_dim, bias=False)

        # regularization
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # check if the flash attention is available - this will make computation fast
        self.flash = hasattr(F, 'scaled_dot_product_attnetion')

    def forward(self, input, context=None):
        h = self.nhead

        q, k, v = (self.to_q(input), *self.to_kv(context).chunk(2, dim=-1))
        q = rearrange(q, "b ... (h d) -> b h ... d", h=h) * self.scale

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        
        else:
            sim = torch.einsum("b ... n d, b t d -> b ... n t", q, k)
            att = self.attn_dropout(sim.softmax(dim=-1))
            y = torch.einsum("b ... n t, b t d -> b ... n d", att, v)

        y = rearrange(y, "b h ... d -> b ... (h d)")

        return self.to_o(y)
    
class MultiAxisAttention(nn.Module):
    """
    Params:
        nchannels: the hidden size of the image input
        block_size: the kernel size of the image blocks
        nhead: number of heads
        dropout: dropout rate
    Args:
       img: the image input (b, n, p, c) 
    Returns:
        y: the final hidden state (b, n, p, c)
    """
    def __init__(self, nchannels:int, nhead:int, dropout:float) -> None:
        super(MultiAxisAttention, self).__init__()
        
        # cache shape informations
        self.nhead = nhead
        self.scale = (nchannels // nhead) ** -0.5

        # projection layers
        self.to_qkv = nn.Linear(nchannels, 3 * nchannels, bias=False)
        self.to_o = nn.Linear(nchannels, nchannels, bias=False)

        # check if the flash attention is available
        self.flash = hasattr(F, 'scaled_dot_product_attnetion')

        # regularization based on flash attention
        if self.flash:
            self.dropout = dropout
        else:
            self.attn_dropout = nn.Dropout(dropout)

    def forward(self, img_blocks):
        # cache the head and channel information
        h = self.nhead

        # project the image into query and key values
        q, k, v = self.to_qkv(img_blocks).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n p (h c) -> b h n p c', h = h), (q, k, v))
        q = q * self.scale

        # split and transpose
        # this will let us perform dilated and regional self attention 
        q, k, v = axis_splitting(q, k, v, h)

        # flash attention for really fast computation
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p = self.dropout if self.training else 0.0, is_causal=False
            )
        # slower operation using einsum
        else:
            sim = torch.einsum("b h n p c, b h n q c -> b h n p q", q, k)
            att = F.dropout(sim.softmax(dim=-1))
            y = torch.einsum("b h n p q, b h n q c -> b h n p c", att, v)

        y = rearrange(y, "b h n p c -> b n p (h c)")

        return y

# helper functions
def img_partitioner(img, block_size, nchannels):
    # assumption checks
    assert img.shape[1] == img.shape[2]
    assert int(img.shape[2] ** 0.5) == block_size

     # create the blocker
    blocker = nn.Unfold(block_size, stride=block_size)

    # chopping the images into blocks
    img_blocks = rearrange(img, "b h w c -> b c h w")
    img_blocks = blocker(img_blocks)
    
    return rearrange(img_blocks, "b (c p) n -> b n p c", c=nchannels)


def axis_splitting(q, k, v, h):
    # channel splitting
    (q1, q2), (k1, k2), (v1, v2) = map(lambda t: (t[:, :h // 2, ...], t[:, h // 2:, ...]), (q, k, v))
    
    # transpose and concat
    q1, k1, v1 = map(lambda t: torch.transpose(t, 2, 3), (q1, k1, v1))
    qt, kt , vt = map(lambda ts: torch.concat(ts, dim=1), ((q1, q2), (k1, k2), (v1, v2)))

    return qt, kt, vt

def get_text_encoder(t5_model):
    model = T5EncoderModel.from_pretrained(t5_model)
    # Freeze the weights
    for param in model.parameters():
        param.requires_grad = False
    return model

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x, approximate='tanh')
    
# layer norm with bias
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)