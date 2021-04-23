# %%
import torch
import torch.nn as nn

encoder_config = {"hooks": [2, 5, 8, 11], "dropout": 0.1, "name": "base"}
reassemble_config = {
    "output_features": 256,
    "features": [256, 512, 768, 768],
    "img_size": [384, 384],
    "s": [4, 8, 16, 32],
    "input_features": 768,
}


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class Reassemble(nn.Module):
    def __init__(self, config=reassemble_config, index=3, use_bn=False):
        super(Reassemble, self).__init__()
        img_size, input_features, features, s, output_features = (
            config["img_size"],
            config["input_features"],
            config["features"][index],
            config["s"][index],
            config["output_features"],
        )
        last_layer = nn.Identity()
        if s < 16:
            last_layer = nn.ConvTranspose2d(features, features, 16 // s, stride=16 // s)
        elif s > 16:
            last_layer = nn.Conv2d(features, features, 3, stride=s // 16, padding=1)
        self.conv = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([img_size[0] // 16, img_size[1] // 16])),
            nn.Conv2d(input_features, features, 1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            last_layer,
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
        )
        self.out = nn.Conv2d(features, output_features, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch size, number of patch + 1, embedding)
        # ignore the class token
        # B, 576, 768
        x = x[:, 1:]
        # B, 384, 384, C
        x = self.conv(x)
        x = self.out(x)
        return x


# RMSE: 0.98
# class Reassemble(nn.Module):
#     def __init__(self, config=reassemble_config, task="depth"):
#         super(Reassemble, self).__init__()
#         self.linear = nn.Linear(768, 256)
#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch size, number of patch + 1, embedding)
#         # ignore the class token
#         batch_size, num_patch, embedding_size = x.shape
#         num_patch -= 1
#         # B, 576, 768
#         x = x[:, 1:]
#         # B, 576, 256
#         x = self.linear(x)
#         # B, 384, 384, C
#         x = x.reshape(batch_size, self.config["img_size"], self.config["img_size"], 1)
#         # B, C, H, W
#         x = x.permute(0, 3, 1, 2)
#         return x.squeeze()


# RMSE:1.1
# class Reassemble(nn.Module):
#     def __init__(self, config=reassemble_config, task="depth"):
#         super(Reassemble, self).__init__()
#         num_class = reassemble_config["num_class"] if task != "depth" else 1
#         self.linear = nn.Linear(3, num_class)
#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch size, number of patch + 1, embedding)
#         # ignore the class token
#         batch_size, num_patch, embedding_size = x.shape
#         num_patch -= 1
#         x = x[:, 1:]
#         # B, H, W, 3
#         x = x.reshape(batch_size, self.config["img_size"], self.config["img_size"], 3)
#         x = self.linear(x)
#         # B, C, H, W
#         x = x.permute(0, 3, 1, 2)
#         return x.squeeze()


if __name__ == "__main__":
    x = torch.randn((2, 577, 768))
    model = Reassemble()
    print(model(x).shape)

# %%
