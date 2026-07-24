import math

import torch
import einops

from torch import nn
from typing import Union



class RoPE(nn.Module):

    def __init__(
            self, 
            theta : float,
            d_k : int,
            max_seq_len : int,
            device : Union[torch.device, None] = None,
            ) -> None:
        
        super().__init__()

        self._theta = theta
        self._d_k = d_k
        self._max_seq_len = max_seq_len
        
        self._device = device

        self.R = torch.zeros(size = (max_seq_len, d_k, d_k), device = device)

        for idx in range(max_seq_len):
            for k in range(1, (d_k//2) + 1):

                exp = ((2 * k) - 2) / d_k
                angle_i_k = idx / (theta**exp)

                idx_1 = 2 * (k - 1) - 1
                idx_2 = 2 * (k - 1)

                self.R[idx, idx_1, idx_1] = math.cos(angle_i_k)
                self.R[idx, idx_1, idx_2] = -math.sin(angle_i_k)
                self.R[idx, idx_2, idx_1] = math.sin(angle_i_k)
                self.R[idx, idx_2, idx_2] = math.cos(angle_i_k)


        self.register_buffer(name = "R_mat", tensor = self.R, persistent = False)
        

    def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:

        # if len(token_positions.shape) > 1:
        #     batch_size, seq_len = token_positions.shape
        # else:
        #     batch_size = 1
        #     seq_len = token_positions.shape[0]


        out : torch.Tensor = torch.zeros(size = x.shape, device = self._device)

        for batch in range(batch_size):
            r_sliced = self.R[token_positions[batch]]
            x_sliced = x[batch, token_positions[batch]]
            print(r_sliced.shape)

            out[batch] = einops.einsum(r_sliced, x_sliced, "seq_len d_k d_k, seq_len d_k -> seq_len d_k")

        return out
    


if __name__ == "__main__":

    batch_size = 5
    seq_len = 10000
    num_vocab = 10000
    d_model = 128

    x = torch.rand( size = (batch_size, seq_len, d_model))
    token_positions = torch.randint(low = 0, high = seq_len, size = (batch_size, seq_len))

    rope = RoPE(theta = 0.5, d_k = d_model, max_seq_len = seq_len)

    out = rope(x, token_positions)

    print(out.shape)