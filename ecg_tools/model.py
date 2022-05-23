from turtle import forward

import einops
import torch
import torch.nn as nn


class LinearEmbedding(nn.Sequential):
    
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__(*[
            nn.Linear(input_channels, output_channels),
            nn.LayerNorm(output_channels),
            nn.GELU()
        ])
        self.cls_token = nn.Parameter(torch.randn(1, output_channels))
        
    def forward(self, x):
        embedded = super().forward(x)
        return torch.cat([einops.repeat(self.cls_token, "n e -> b n e", b=x.shape[0]), embedded], dim=1)
    
    
class MLP(nn.Sequential):
    def __init__(self, input_channels, expansion=4):
        super().__init__(*[
            nn.Linear(input_channels, input_channels * expansion),
            nn.GELU(),
            nn.Linear(input_channels * expansion, input_channels)
        ])
        

if __name__ == "__main__":
    print(LinearEmbedding(3, 192)(torch.rand(2, 128, 3)).shape)
    print(MLP(3)(torch.rand(2, 128, 3)).shape)
        