from typing import Union

import torch

from torch import inf, nn

from transformer.rope import RoPE
from transformer.scaled_dot_product import ScaledDotProduct
from transformer.linear import Linear


class MultiHead(nn.Module):

    def __init__(
            self,
            ) -> None:
        
        super().__init__()

        self._scaled_dot_product = ScaledDotProduct()

    def forward(self, Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor) -> torch.Tensor:

        batch_size, num_heads, seq_len, d_k, = Q.shape
        _, _, _, d_v = V.shape

        attn_scores_multi = torch.zeros(size = (batch_size, num_heads, seq_len, d_v))

        masks = torch.triu(torch.ones(size = (seq_len, seq_len)), diagonal = 1)
        masks = torch.where(masks == 0, True, False)
        print(f"masks : {masks.shape}")

        attn_scores_multi = self._scaled_dot_product(Q, K, V, masks)

        return attn_scores_multi

class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            d_model : int,
            num_heads : int,
            max_seq_len : Union[int, None] = None,
            theta : Union[float, None] = None,
            ) -> None:
        
        super().__init__()

        self._d_model = d_model
        self._num_heads = num_heads

        self._theta = theta
        self._max_seq_len = max_seq_len
        self._rope : Union[RoPE, None] = None


        self._d_k, self._d_v = self._d_model // self._num_heads , self._d_model // self._num_heads

        self._wq = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_k))
        self._wk = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_k))
        self._wv = Linear(in_features = self._d_model , out_features = (self._num_heads * self._d_v))
        self._wo = Linear(in_features = (self._num_heads * self._d_v), out_features = self._d_model)

        if self._theta is not None and self._max_seq_len is not None:

            self._rope = RoPE(theta = self._theta, d_k = self._d_k, max_seq_len = self._max_seq_len)

        self._multihead = MultiHead()

    def forward(self, x : torch.Tensor, tok_pos : Union[torch.Tensor, None] = None) -> torch.Tensor:

        batch_size, seq_len, d_model = x.shape

        attn_scores = torch.Tensor()

        q : torch.Tensor = self._wq(x)
        q = q.view(size = (batch_size, seq_len, self._num_heads, -1))
        q = q.transpose(1, 2)

        if self._rope is not None:

            q = self._rope(q.contiguous().view(size = (batch_size * self._num_heads, seq_len,  -1)), tok_pos)
            q = q.view(size = (batch_size, self._num_heads, seq_len, -1))

        print(f"q : {q.shape}")

        k : torch.Tensor = self._wk(x)
        k = k.view(size = (batch_size, seq_len, self._num_heads, -1))
        k = k.transpose(1, 2)

        if self._rope is not None:

            k = self._rope(k.contiguous().view(size = (batch_size * self._num_heads, seq_len,  -1)), tok_pos)
            k = k.view(size = (batch_size, self._num_heads, seq_len, -1))

        print(f"k : {k.shape}")

        v : torch.Tensor = self._wv(x)
        v = v.view(size = (batch_size, seq_len, self._num_heads, -1))
        v = v.transpose(1, 2)
        print(f"v : {v.shape} ")

        attn_scores_multi : torch.Tensor = self._multihead(q, k, v)
        attn_scores_multi = attn_scores_multi.transpose(1, 2)
        print(f"attn_scores_multi : {attn_scores_multi.shape}")

        attn_scores = attn_scores_multi.contiguous().view(batch_size, seq_len, -1)
        print(f"attn_scores : {attn_scores.shape}")

        return self._wo(attn_scores)
    

if __name__ == "__main__":

    batch_size = 2
    seq_len = 256
    d_model = 64
    num_heads = 4

    tok_pos = torch.randint(low = 0, high = seq_len, size = (1, seq_len)).squeeze(0)

    theta = 10000

    x = torch.rand(size = (batch_size, seq_len, d_model))

    multi_head_self_attention = MultiHeadSelfAttention(d_model = d_model, num_heads = num_heads, max_seq_len = seq_len, theta= theta)

    out = multi_head_self_attention(x, tok_pos)

    print(out.shape)