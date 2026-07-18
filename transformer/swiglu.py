import torch
import einops

from torch import nn
from typing import Union

from transformer.linear import Linear



class SwiGLU(nn.Module):

    def __init__(
            self, 
            d_model : int,
            d_ff : int,
            device : Union[torch.device, None] = None,
            dtype : Union[torch.dtype, None] = None
            ) -> None:
        
        super().__init__()

        self._d_model = d_model
        self._d_ff = d_ff
        
        self._device = device
        self._dtype = dtype

        self.linear_1 = Linear(in_features = self._d_model, out_features = self._d_ff, device = device, dtype = dtype)
        self.linear_3 = Linear(in_features = self._d_model, out_features = self._d_ff, device = device, dtype = dtype)
        self.linear_2 = Linear(in_features = self._d_ff, out_features = self._d_model, device = device, dtype = dtype)
        

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, d_model = x.shape

        out : torch.Tensor = torch.zeros(size = (batch_size, seq_len, self._d_model), dtype = self._dtype, device = self._device)

        out_linear_1 = self.linear_1(x)
        out_linear_3 = self.linear_3(x)

        out_silu = out_linear_1 * torch.sigmoid(out_linear_1)
        out = self.linear_2(out_silu * out_linear_3)

        return out
    


if __name__ == "__main__":

    batch_size = 5
    seq_len = 10000
    num_vocab = 10000
    d_model = 128
    d_ff = int((8/3) * d_model)

    x = torch.rand( size = (batch_size, seq_len, d_model))

    swi_glu = SwiGLU(d_model = d_model, d_ff = d_ff)

    out = swi_glu(x)

    print(out.shape)