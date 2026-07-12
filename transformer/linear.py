import torch
import einops

from torch import nn
from typing import Union



class Linear(nn.Module):

    def __init__(
            self, 
            in_features : int,
            out_features : int ,
            device : Union[torch.device, None] = None,
            dtype : Union[torch.dtype, None] = None
            ) -> None:
        
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        
        self._device = device
        self._dtype = dtype

        self._std = (2 / (self._in_features + self._out_features))

        self._weights = nn.Parameter(torch.normal(mean = 0,
                                      std = self._std,
                                      size = (self._out_features, self._in_features),
                                      dtype = self._dtype,
                                      device = self._device))
        
        self._weights_truncated = nn.init.trunc_normal_(self._weights, a = -3 * self._std, b = 3 * self._std)

        self.w1 = self._weights_truncated


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        out : torch.Tensor = torch.Tensor()

        out = einops.einsum(self.w1, x, "d_out d_in, ... d_in -> ... d_out")

        return out
    


if __name__ == "__main__":

    x = torch.rand(size = (4, 3))

    linear = Linear(4, 5)

    out = linear(x)

    print(out.shape)