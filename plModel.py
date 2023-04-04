import torch
import torch.nn as nn
import numpy as np
import einops
import math

import pytorch_lightning as pl


class PositionalEncoding(pl.LightningModule):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 784):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class MaskedSelfAttention(pl.LightningModule):
    def __init__(self, dim= 512, num_classes= 256):
        super(MaskedSelfAttention, self).__init__()

        self.dim       = dim

        # To convert single dim images to multidim
        # img[N, H*W] -> R[N, H*W, embedding_dim]

        self.Q         = nn.Linear(self.dim, self.dim)
        self.K         = nn.Linear(self.dim, self.dim)
        self.V         = nn.Linear(self.dim, self.dim)

        self.linear    = nn.LazyLinear(self.dim, bias= False)
        self.scaling_factor = torch.tensor(self.dim**-0.5)

    def forward(self, x, mask=None):
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        scaled_dot = torch.einsum('b i d, b j d-> b i j', q, k) * self.scaling_factor

        if mask is not None:
            scaled_dot = scaled_dot.masked_fill(mask == 0, -np.inf)

        look_ahead_mask = torch.tril(torch.ones(x.size(0), scaled_dot.size(-1), scaled_dot.size(-2)), 0)
        look_ahead_mask = look_ahead_mask.to(scaled_dot.device)
        
        scaled_dot = scaled_dot.masked_fill(look_ahead_mask == 0, -np.inf)
        attention = torch.softmax(scaled_dot, dim=-1)
        attention = torch.einsum('b i j,  b j d -> b i d', attention, v)

        out = attention
        out = self.linear(out)

        # Rearranging for nn.CrossEntropyLoss format
        # out = einops.rearrange(out ,'b s d -> b d s')
        return out

    def synthesize(self, x, img_size = 28):
        
        x_l  = x.size(-1) # Seq Length
        start_index = x_l//28

        x    = x[:, 0:start_index]
        x    = torch.cat((x, torch.zeros((x.size(0), x_l - start_index), device=x.device)), dim =-1)
        mask = torch.full((1, x_l - start_index), 0)
        mask = torch.cat((torch.ones(x.size(0), start_index), mask), dim= -1)
        mask = mask.to(x.device)


        t = range(x_l)
        t = t[start_index:]

        for i in t:
            pred       = self.forward(x.long(), mask)
            pred       = einops.rearrange(pred, 'b i j -> b j i')
            mask[:, i] = 1
            x[:, i]    = torch.argmax(pred[:, i])
        
        x = einops.rearrange(x, 'b (h w) -> b h w', h=img_size, w=img_size).detach().cpu().numpy()
        return x


class MultiHeadedAttention(pl.LightningModule):
    def __init__(self, heads= 3, dim= 512):
        super(MultiHeadedAttention, self).__init__()

        self.heads  = heads

        # For now, not splitting heads...
        # It is essentially the same thing requring more compute 
        
        self.MHA    = nn.ModuleList([MaskedSelfAttention(dim) for i in range(heads)])
        self.linear = nn.LazyLinear(dim)


    def forward(self, x):

        MHA_l = []
        for Attention in self.MHA:
            MHA_l.append(Attention(x))
        
        x = torch.cat(MHA_l, dim= -1)
        x = self.linear(x)

        # -- x should have the same dim as the original
        # -- x[N, SeqLen, EmbedDim] 

        return x
        

class TransformerBlock(pl.LightningModule):
    def __init__(self, heads= 3, dim= 512, num_classes= 4):
        super(TransformerBlock, self).__init__()

        self.MHA = MultiHeadedAttention(heads= heads, dim=dim)
        self.ffn = nn.LazyLinear(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        

        x = x + self.norm1(self.MHA(x))
        x = x + self.norm2(self.ffn(x))
        
        return x


class DecoderStack(pl.LightningModule):
    def __init__(self, layers = 6, heads= 3, dim= 512, num_classes= 4):
        super(DecoderStack, self).__init__()


        self.input_embedding = nn.Embedding(num_classes, 64)
        self.pe              = PositionalEncoding(dim, 0.1)

        self.t_blocks = nn.ModuleList(TransformerBlock(heads = heads, dim= dim, num_classes=num_classes) for i in range(layers))
        self.linear   = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.input_embedding(x)
        x = self.pe(x)

        for layer in self.t_blocks:
            x = layer(x)

        x  = self.linear(x)
        return x





# Strictly for testing purposes
if __name__ == "__main__":
    crit = nn.CrossEntropyLoss()
    dim = 64
    seq = 784
    bs  = 2
    num_classes = 256
    # MSA = MaskedSelfAttention(dim=dim)
    # MHA = MultiHeadedAttention(dim=dim)
    # tb  = TransformerBlock(heads = 3, dim = dim, num_classes=num_classes)

    ds  = DecoderStack(heads = 3, dim = dim, num_classes=num_classes)

    t  = torch.randint(high=num_classes, size=(bs//2, seq))

    out   = ds(t)

    # np.save('/home/jarvis/Projects/ar_image/results/t.npy', out)
    # loss = crit(out, t)





