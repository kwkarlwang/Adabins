# %%
import torch
import torch.nn as nn
from .encoder import Encoder
from .reassemble import Reassemble


class Transformer_Reassemble(nn.Module):
    def __init__(self, config: dict, task="depth"):
        super().__init__()
        self.encoder = Encoder(config["encoder"])
        self.reassemble = Reassemble(config["reassemble"], task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x)
        return self.reassemble(outputs[11])


if __name__ == "__main__":
    model = Transformer_Reassemble()
    x = torch.randn((2, 3, 384, 384))
    print(model(x).shape)

# %%
