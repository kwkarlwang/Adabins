import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size=10, embedding_dim=128, num_heads=4):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=1024
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)

        self.embedding_convPxP = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.positional_encodings = nn.Parameter(
            torch.rand(500, embedding_dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        # shape = batch, embedding_dim, num_tokens
        embeddings: torch.Tensor = self.embedding_convPxP(x).flatten(2)
        embeddings: torch.Tensor = embeddings + self.positional_encodings[
            : embeddings.shape[2], :
        ].T.unsqueeze(0)

        # shape = num tokens, batch, embedding dim
        embeddings = embeddings.permute(2, 0, 1)
        # shape = num_tokens, batch, embedding dim
        x = self.transformer_encoder(embeddings)
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, K: torch.Tensor):
        batch, channel, height, width = x.size()
        # transformer output
        _, cout, channelk = K.size()
        assert (
            channel == channelk
        ), "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"

        y = torch.matmul(
            x.view(batch, channel, height * width).permute(0, 2, 1), K.permute(0, 2, 1)
        )
        return y.permute(0, 2, 1).view(batch, cout, height, width)
