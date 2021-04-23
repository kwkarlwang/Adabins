import torch
import torch.nn as nn


class ResidualConvolutionUnit(nn.Module):
    def __init__(self, features: int, use_bn=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return x + out


class Fusion(nn.Module):
    def __init__(self, features: int, use_bn=False):
        super().__init__()
        self.rcu1 = ResidualConvolutionUnit(features, use_bn)
        self.rcu2 = ResidualConvolutionUnit(features, use_bn)
        output_features = features
        self.conv = nn.Sequential(
            nn.Conv2d(features, output_features, 1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(output_features) if use_bn else nn.Identity(),
        )

    def forward(self, x_out, x_in=None):
        if x_in is not None:
            x_out += self.rcu1(x_in)
        x_out = self.rcu2(x_out)
        x_out = nn.functional.interpolate(
            x_out, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_out = self.conv(x_out)

        return x_out


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        features = config["features"]
        self.fusion1 = Fusion(features)
        self.fusion2 = Fusion(features)
        self.fusion3 = Fusion(features)
        self.fusion4 = Fusion(features)

    def forward(self, xs: list):
        x = self.fusion1(xs[0])
        x = self.fusion2(x, xs[1])
        x = self.fusion3(x, xs[2])
        x = self.fusion4(x, xs[3])
        return x