import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import (
    block_size,
    batch_size,
    n_embed,
    n_head,
    device,
    n_layer,
    dropout,
)



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    
    def forward(self, ix):
        B,T,C = ix.shape
        k = self.key(ix)
        q = self.query(ix)

        wei = k @ q.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(ix)

        out = wei @ v
        return out
    

class MultiHead(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.heads = nn.Sequential(*[Head(n_embed // n_head) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ix):
        x = torch.cat([head(ix) for head in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.GELU(),
            nn.Linear(n_embed*4, n_embed*8),
            nn.GELU(),
            nn.Linear(n_embed*8, n_embed),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x




class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead(n_head)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    
    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        out = x + self.ff(self.ln2(x))

        return out


class Transformer(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])

        self.ln = nn.LayerNorm(n_embed)
        self.linear = nn.Linear(n_embed, self.vocab_size)
        self.tokenizer = tokenizer
    
    def forward(self, ix, targets=None):
        B,T = ix.shape
        token_embed =  self.token_embedding(ix)
        pos_embed = self.position_embedding(torch.arange(T, device=device))

        logits = token_embed + pos_embed
        logits = self.blocks(logits)

        logits = self.ln(logits)
        logits  = self.linear(logits)

        if targets == None:
            loss = None
        
        else:
            B,T,C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def predict(self, context):
        ix = context[:, -block_size:]
        logits, _ = self(ix)
        logits = F.softmax(logits, dim=-1)
        logits =logits[:, -1, :]
        ix_next = torch.multinomial(logits, num_samples=1)
        return ix_next

    

    def generate(self, ix, max_token):
        for _ in range(max_token):
            ix_next = self.predict(ix)
            new_text = self.tokenizer.decode(ix_next[0].tolist())
            print(new_text[0], end=" ")
            # with open('output.txt', 'a') as f:
            #     f.write(new_text)

            ix = torch.cat((ix, ix_next), dim=1)
        return ix
