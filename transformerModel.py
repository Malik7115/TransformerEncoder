import torch
import torch.nn as nn
import numpy as np
import einops
import math

from torch.utils.data import DataLoader
import einops
from cDataloader import tokenized_datasets, data_collator

# from cDataloader import numDataset


class PositionalEncoding(nn.Module):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    '''

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
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

class LearnablePositionEncoding(nn.Module):
    def __init__(self, dim= 512, vocab_size= 256):
        super().__init__()

        self.position_embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        pe = self.position_embedding(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        return x + pe
        

class MaskedSelfAttention(nn.Module):
    def __init__(self, dim= 512, causal_mask= None):
        super(MaskedSelfAttention, self).__init__()

        self.dim         = dim
        self.causal_mask = causal_mask

        # To convert single dim images to multidim
        # img[N, H*W] -> R[N, H*W, embedding_dim]

        self.Q         = nn.Linear(self.dim, self.dim)
        self.K         = nn.Linear(self.dim, self.dim)
        self.V         = nn.Linear(self.dim, self.dim)

        self.linear    = nn.LazyLinear(self.dim, bias= False)
        self.scaling_factor = torch.sqrt(torch.tensor(self.dim))

    def forward(self, x, mask=None):
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        scaled_dot = torch.einsum('b i d, b j d-> b i j', q, k) / self.scaling_factor

        if mask is not None:
            mask = einops.repeat(mask, 'b c -> b r c', r = scaled_dot.size(-1))
            scaled_dot = scaled_dot.masked_fill(mask == 0, -np.inf)

        if self.causal_mask is not None:
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

    # def synthesize(self, x, img_size = 28):
        
    #     x_l  = x.size(-1) # Seq Length
    #     start_index = x_l//28

    #     x    = x[:, 0:start_index]
    #     x    = torch.cat((x, torch.zeros((x.size(0), x_l - start_index), device=x.device)), dim =-1)
    #     mask = torch.full((1, x_l - start_index), 0)
    #     mask = torch.cat((torch.ones(x.size(0), start_index), mask), dim= -1)
    #     mask = mask.to(x.device)


    #     t = range(x_l)
    #     t = t[start_index:]

    #     for i in t:
    #         pred       = self.forward(x.long(), mask)
    #         pred       = einops.rearrange(pred, 'b i j -> b j i')
    #         mask[:, i] = 1
    #         x[:, i]    = torch.argmax(pred[:, i])
        
    #     x = einops.rearrange(x, 'b (h w) -> b h w', h=img_size, w=img_size).detach().cpu().numpy()
    #     return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads= 3, dim= 512):
        super(MultiHeadedAttention, self).__init__()

        self.heads  = heads

        # For now, not splitting heads...
        # It is essentially the same thing requring more compute 
        
        self.MHA    = nn.ModuleList([MaskedSelfAttention(dim) for i in range(heads)])
        self.linear = nn.LazyLinear(dim)


    def forward(self, x, mask=None):

        MHA_l = []
        for Attention in self.MHA:
            MHA_l.append(Attention(x, mask))
        
        x = torch.cat(MHA_l, dim= -1)
        x = self.linear(x)

        # -- x should have the same dim as the original
        # -- x[N, SeqLen, EmbedDim] 

        return x
        

class TransformerBlock(nn.Module):
    def __init__(self, heads= 3, dim= 512):
        super(TransformerBlock, self).__init__()

        self.MHA = MultiHeadedAttention(heads= heads, dim=dim)


        # This is where the real memorization happens
        # it is sort of an overunder autoencoder
        self.ffn = nn.Sequential(
            nn.LazyLinear(4*dim),
            nn.GELU(),
            nn.LazyLinear(dim),
            nn.Dropout(0.2)
        )
        # self.ffn = nn.LazyLinear(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        


    def forward(self, x, mask = None):
        
        x = x + self.norm1(self.MHA(x, mask))
        x = x + self.norm2(self.ffn(x))
        
        return x


class EncoderStack(nn.Module):
    def __init__(self, layers = 6, heads= 3, dim= 64, vocab_size= 50000, num_classes=2, learnable_pe= True):
        super(EncoderStack, self).__init__()

        self.learnable_pe     = learnable_pe
        self.input_embedding  = nn.Embedding(vocab_size, dim)
        self.rpe              = PositionalEncoding(dim, 0.1)
        self.lpe              = LearnablePositionEncoding(dim=dim, vocab_size=vocab_size)

        self.t_blocks = nn.ModuleList(TransformerBlock(heads = heads, dim= dim) for i in range(layers))
        self.linear   = nn.LazyLinear(dim)


        self.mlp      = nn.Sequential(nn.LazyLinear(dim//2), nn.LazyLinear(num_classes))

    def forward(self, x, mask=None):

        x = self.input_embedding(x) # x = [B, seq_len, dim]

        if (self.learnable_pe == True):
            x = self.lpe(x)
        
        else:
            x = self.rpe(x.permute(1,0,2))
            x = x.permute(1,0,2)

        for layer in self.t_blocks:
            x = layer(x, mask)

        """

        The input and output shape should be the same
        for clasification tasks, add another mlp head
        depending on the task, it might look like this:
        input = [B, seq_len, dim], output = [B, seq_len, num_classes]

        Usually for classification tasks one might use only one of the tokens
        
        """

        x  = self.linear(x) # x = [B, seq_len, dim]

        # Use the classsification head with only the CLS token
        x  = self.mlp(x[:,0,:])


        return x





# Strictly for testing purposes
if __name__ == "__main__":

    import os
    os.system('clear')

################################################################################


    crit = nn.CrossEntropyLoss()
    dim = 64
    seq = 784
    bs  = 2
    num_classes = 2


################################################################################

    train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
)

    test_dataloader = DataLoader(
        tokenized_datasets["test"], shuffle=True, batch_size=2, collate_fn=data_collator
)


    ds    = EncoderStack(num_classes=num_classes)

    batch = next(iter(train_dataloader))
    labels     = batch['labels']
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    out   = ds(input_ids, mask=attention_mask)
    loss  = crit(out, labels)

    print(loss.size())





