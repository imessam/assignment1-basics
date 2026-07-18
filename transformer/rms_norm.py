import torch

from torch import nn
from typing import Union



class RMSNorm(nn.Module):

    def __init__(
            self, 
            d_model : int,
            eps : float = 1e-5 ,
            device : Union[torch.device, None] = None,
            dtype : Union[torch.dtype, None] = None
            ) -> None:
        
        super().__init__()

        self._d_model = d_model
        self._eps = eps
        
        self._device = device
        self._dtype = dtype

        self._weights = nn.Parameter(torch.ones(
                                      self._d_model,
                                      dtype = self._dtype,
                                      device = self._device))
        
        self.g = self._weights


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, d_model = x.shape

        out : torch.Tensor = torch.zeros(size = (batch_size, seq_len, self._d_model), dtype = self._dtype, device = self._device)

        in_type = x.dtype

        x = x.to(torch.float32)

        x_sq = torch.pow(x, 2)

        x_rms = torch.sqrt((torch.sum(x_sq, dim = 2, keepdim = True) / self._d_model) + self._eps)

        out = (x / x_rms) * self.g

        out = out.to(in_type)

        return out
    


if __name__ == "__main__":

    batch_size = 5
    seq_len = 10000
    num_vocab = 10000
    d_model = 128

    x = torch.randint(low = 0, high = 10000, size = (batch_size, seq_len, d_model))

    rms_norm = RMSNorm(d_model = num_vocab)

    out = rms_norm(x)

    print(out.shape)