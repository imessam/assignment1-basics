import torch
import einops

from torch import nn
from typing import Union



class Embedding(nn.Module):

    def __init__(
            self, 
            num_embeddings : int,
            embeddings_dim : int ,
            device : Union[torch.device, None] = None,
            dtype : Union[torch.dtype, None] = None
            ) -> None:
        
        super().__init__()

        self._num_embeddings = num_embeddings
        self._embeddings_dim = embeddings_dim
        
        self._device = device
        self._dtype = dtype

        self._weights = nn.Parameter(torch.normal(mean = 0,
                                      std = 1,
                                      size = (self._num_embeddings, self._embeddings_dim),
                                      dtype = self._dtype,
                                      device = self._device))
        
        self._weights_truncated = nn.init.trunc_normal_(self._weights, a = -3 , b = 3 )

        self.w = self._weights_truncated


    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = token_ids.shape

        out : torch.Tensor = torch.zeros(size = (batch_size, seq_len, self._embeddings_dim), dtype = self._dtype, device = self._device)

        for idx in range(batch_size):

            out[idx] = self.w[token_ids[idx]]

        return out
    


if __name__ == "__main__":

    batch_size = 5
    seq_len = 10000
    num_vocab = 10000
    d_embed = 128

    token_ids = torch.randint(low = 0, high = 10000, size = (batch_size, seq_len))

    embedding = Embedding(num_embeddings = num_vocab, embeddings_dim = d_embed)

    out = embedding(token_ids)

    print(out.shape)