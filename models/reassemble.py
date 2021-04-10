# %%
import torch
import torch.nn as nn

reassemble_config = {"num_class": 41, "img_size": 384}


class Reassemble(nn.Module):
    def __init__(self, config=reassemble_config, task="depth"):
        super(Reassemble, self).__init__()
        num_class = reassemble_config["num_class"] if task != "depth" else 1
        self.linear = nn.Linear(3, num_class)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch size, number of patch + 1, embedding)
        # ignore the class token
        batch_size, num_patch, embedding_size = x.shape
        num_patch -= 1
        x = x[:, 1:]
        # B, H, W, 3
        x = x.reshape(batch_size, self.config["img_size"], self.config["img_size"], 3)
        x = self.linear(x)
        # B, C, H, W
        x = x.permute(0, 3, 1, 2)
        return x.squeeze()


if __name__ == "__main__":
    x = torch.randn((2, 577, 768))
    model = Reassemble()
    print(model(x).shape)

# %%
