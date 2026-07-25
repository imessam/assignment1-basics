import math

import torch
import einops

from torch import inf, nn
from typing import Union

from transformer.softmax import Softmax

class ScaledDotProduct(nn.Module):

    def __init__(
            self
            ) -> None:
        
        super().__init__()

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : Union[None, torch.Tensor] = None) -> torch.Tensor:

        d_k = q.shape[-1]

        attn_weights = einops.einsum(q, k, 
                                     "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") 
        attn_weights = attn_weights / math.sqrt(d_k)
        # print(attn_weights.shape)

        if mask is not None:
            masks = torch.where(mask == False, -inf, 0)
            # print(masks.shape)

            attn_weights += masks
            # print(attn_weights.shape)

        attn_weights_probs = Softmax()(attn_weights, dim = -1)
        # print(attn_weights_probs.shape)

        attn_scores = einops.einsum(attn_weights_probs, v, 
                                     "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")
        
        # print(attn_scores.shape)

        return attn_scores
    


if __name__ == "__main__":

    batch_size = 2
    num_q = 2
    seq_len = 256
    d_k = 64
    d_v = 64

    q = torch.rand( size = (batch_size, num_q, d_k))
    k = torch.rand( size = (batch_size, seq_len, d_k))
    v = torch.rand( size = (batch_size, seq_len, d_v))
    masks = torch.randint(low = 0, high = 2, size = (num_q, seq_len))

    scaled_dot_product = ScaledDotProduct()

    out = scaled_dot_product(q, k, v, masks)

    print(out.shape)