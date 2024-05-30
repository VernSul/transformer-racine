import sys
import tiktoken
import torch
from transformer import Transformer
from config import (
    device
)

tokenizer = tiktoken.encoding_for_model("gpt-4")


file_name = sys.argv[1]
max_tokens = sys.argv[2]

print(file_name, max_tokens)

model = Transformer()
model.to(device)



def predict(max_tokens=5000):
    tokens = int(max_tokens)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    context = torch.ones((1,1), dtype=torch.long, device=device)

    model.generate(context, max_token=tokens)

if __name__ == "__main__":
    predict(max_tokens)

