import torch

from torch import nn
from typing import Union

from transformer.linear import Linear



class Softmax(nn.Module):

    def __init__(
            self
            ) -> None:
        
        super().__init__()

    def forward(self, x : torch.Tensor, dim : int) -> torch.Tensor:

        out : torch.Tensor = torch.zeros_like(x)

        max_val = torch.max(x, dim = dim, keepdim = True)
        # print(max_val.indices.shape, max_val.values.shape)

        x_sub = x - max_val.values
        # print(x_sub.shape)

        out = torch.softmax(x_sub, dim = dim)
        # print(out.shape)

        return out
    


if __name__ == "__main__":

    batch_size = 2
    num_vocab = 10000

    x = torch.rand( size = (batch_size, num_vocab))

    softmax = Softmax()

    out = softmax(x, dim = 1)

    print(out.shape)