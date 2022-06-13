from math import sqrt

import einops
from einops.layers.torch import Reduce
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


class ResidualAdd(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, x):
        assert len(x.shape) == 3
        keys = self.keys_projection(x)
        values = self.values_projection(x)
        queries = self.queries_projection(x)
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)


class TransformerEncoderLayer(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4, num_heads=8, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__(
            *[
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MultiHeadAttention(embed_size, num_heads),
                    nn.Dropout(dropout)
                ])),
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MLP(embed_size, expansion),
                    nn.Dropout(dropout)
                ]))
            ]
        )


class Classifier(nn.Sequential):
    def __init__(self, embed_size, num_classes):
        super().__init__(*[
            Reduce("b n e -> b e", reduction="mean"),
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        ])


class ECGformer(nn.Module):

    def __init__(self, num_layers, signal_length, num_classes, input_channels, embed_size, num_heads, expansion) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([TransformerEncoderLayer(
            embed_size=embed_size, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)])
        self.classifier = Classifier(embed_size, num_classes)
        self.positional_encoding = nn.Parameter(torch.randn(signal_length + 1, embed_size))
        self.embedding = LinearEmbedding(input_channels, embed_size)

    def forward(self, x):
        embedded = self.embedding(x)

        for layer in self.encoder:
            embedded = layer(embedded + self.positional_encoding)

        return self.classifier(embedded)


if __name__ == "__main__":
    print(LinearEmbedding(3, 192)(torch.rand(2, 128, 3)).shape)
    print(MLP(3)(torch.rand(2, 128, 3)).shape)
    print(TransformerEncoderLayer(192, 8)(torch.rand(2, 128, 192)).shape)
    print(ECGformer(
        4, 128, 5, 3, 192, 8, 4
    )(torch.rand(2, 128, 3)).shape)
