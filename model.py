import torch
from torch import nn

from tokenizers.USE import USEncoder


class RT1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.USEncoder = USEncoder()
        