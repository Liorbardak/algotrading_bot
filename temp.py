import torch
import torch.nn as nn

embed_dim = 64       # D_model
num_heads = 8        # Number of attention heads
seq_len = 10         # Length of input sequence
batch_size = 4

# Random input tensor: (seq_len, batch_size, embed_dim)
x = torch.rand(seq_len, batch_size, embed_dim)
y = torch.rand(seq_len//2, batch_size, embed_dim)

mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

# attention  K = V = x
#                    Q = y
attn_output, attn_weights = mha(y, x, x)

print(attn_output.shape)   # (seq_len, batch_size, embed_dim)
print(attn_weights.shape)  # (batch_size, seq_len//2, seq_len)

mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
x = torch.rand(batch_size, seq_len, embed_dim)
y = torch.rand(batch_size, seq_len//2, embed_dim)
attn_output, attn_weights = mha(x, x, x)