import math

import torch
import einops

from torch import inf, nn
from typing import Union

from transformer.rope import RoPE
from transformer.scaled_dot_product import ScaledDotProduct
from transformer.linear import Linear

class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            d_model : int,
            num_heads : int,
            theta : int,
            max_seq_len : int,
            ) -> None:
        
        super().__init__()

        self._d_model = d_model
        self._num_heads = num_heads
        self._theta = theta
        self._max_seq_len = max_seq_len

        self._d_k, self._d_v = self._d_model // self._num_heads , self._d_model // self._num_heads

        self._wq = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_k))
        self._wk = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_k))
        self._wv = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_v))
        self._wo = Linear(in_features = (self._num_heads * self._d_v), out_features = self._d_model)

        self._rope = RoPE(theta = self._theta, d_k = self._d_k, max_seq_len = self._max_seq_len)
        self._scaled_dot_product = ScaledDotProduct()

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, d_model = x.shape

        attn_scores = torch.Tensor()

        q : torch.Tensor = self._wq(x)
        q_h = q.view(size = (self._num_heads, batch_size, seq_len, -1))
        print(f"q : {q.shape} , q_h : {q_h.shape}")

        k = self._wk(x)
        k_h = k.view(size = (self._num_heads, batch_size, seq_len, -1))
        print(f"k : {k.shape} , k_h : {k_h.shape}")

        v = self._wv(x)
        v_h = v.view(size = (self._num_heads, batch_size, seq_len, -1))
        print(f"v : {v.shape} , v_h : {v_h.shape}")

        return attn_scores
    


if __name__ == "__main__":

    batch_size = 2
    seq_len = 256
    d_model = 64
    num_heads = 4
    theta = 10000


    x = torch.rand( size = (batch_size, seq_len, d_model))

    multi_head_self_attention = MultiHeadSelfAttention(d_model = d_model, num_heads = num_heads, theta = theta, max_seq_len = seq_len)

    out = multi_head_self_attention(x)

    print(out.shape)