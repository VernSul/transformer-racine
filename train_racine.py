import time

import tiktoken
import torch
import torch.nn as nn
from transformer import Transformer
from config import (
    block_size,
    device,
    learning_rate,
    eval_iter,
    max_iter,
    batch_size,
    n_layer,
    n_embed,
    lr_1,
    lr_2,
    chg_lr_iter
    
)

tokenizer = tiktoken.encoding_for_model("gpt-4")

text = open('data/racine_agg.txt').read()

print(f'Data set has {len(tokenizer.encode(text))} tokens')



encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)


n = int(0.9 * len(encoded))


train_data = encoded[:n]
test_data = encoded[n:]


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    data = train_data if split == 'train' else test_data
    rand = torch.randint(0, len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in rand])
    Y = torch.stack([data[i+1:i+block_size+1] for i in rand])
    X, Y = X.to(device), Y.to(device)
    return X, Y


model = Transformer()
model.to(device)


param_num = sum(p.numel() for p in model.parameters())/1e6

print(param_num, 'M parameters')
model_file = f'racine_lr{learning_rate}_layer{n_layer}_nemb{n_embed}_{param_num:.0f}M'

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_1)
start_time = time.time()

for iter in range(max_iter):
    if iter == chg_lr_iter:
        optimizer = torch.optim.AdamW(model.parameters(), lr_2)
    if iter % eval_iter == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} duration {(time.time() - start_time):.2f} seconds")
        torch.save(model.state_dict(), model_file)
    
    Xb, Yb = get_batch('train')
    logits, loss = model(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), model_file)


