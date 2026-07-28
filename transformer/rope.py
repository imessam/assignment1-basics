import torch

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

        self.pos = torch.arange(max_seq_len)

        self.expo = (2 * torch.arange(0, d_k // 2)) / d_k

        self.angles = torch.zeros(size = (max_seq_len, d_k // 2))

        for pos in self.pos:
            self.angles[pos] = pos / (theta ** self.expo)


        self.sin = torch.sin(self.angles)
        self.cos = torch.cos(self.angles)

    

    def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, d_k =  x.shape

        out : torch.Tensor = torch.zeros(size = x.shape, device = self._device)

        for batch in range(batch_size):
            for pos in token_positions:
    
                x_i = x[batch, pos]
                cos_i = self.cos[pos]
                sin_i = self.sin[pos]

                for k in range(0, d_k // 2):

                    idx_1 = 2 * k
                    idx_2 = idx_1 + 1

                    out[batch, pos, idx_1] = (cos_i[k] * x_i[idx_1]) - (sin_i[k] * x_i[idx_2])
                    out[batch, pos, idx_2] = (sin_i[k] * x_i[idx_1]) + (cos_i[k] * x_i[idx_2])

        return out
    


if __name__ == "__main__":

    batch_size = 5
    seq_len = 10000
    num_vocab = 10000
    d_model = 128

    x = torch.rand( size = (batch_size, seq_len, d_model))
    token_positions = torch.randint(low = 0, high = seq_len, size = (1, seq_len)).squeeze(0)

    rope = RoPE(theta = 0.5, d_k = d_model, max_seq_len = seq_len)

    out = rope(x, token_positions)

    print(out.shape)