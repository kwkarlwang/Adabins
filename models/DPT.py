# %%
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .reassemble import Reassemble

# %%
reassemble_config = {
    "output_features": 256,
    "features": [256, 512, 768, 768],
    "img_size": [384, 384],
    "s": [4, 8, 16, 32],
    "input_features": 768,
}

encoder_config = {"hooks": [2, 5, 8, 11], "dropout": 0.1, "name": "base"}
decoder_config = {"features": 256}
config = {
    "encoder": encoder_config,
    "reassemble": reassemble_config,
    "decoder": decoder_config,
}


class DPT(nn.Module):
    def __init__(self, config=config, head=None):
        super().__init__()
        self.encoder = Encoder(config["encoder"])
        self.reassembles = nn.ModuleList(
            [
                Reassemble(config["reassemble"], i)
                for i in range(len(config["encoder"]["hooks"]))
            ]
        )
        self.decoder = Decoder(config["decoder"])
        self.head = head

    def forward(self, x: torch.Tensor):
        # deeper layer outputs in the end
        xs = self.encoder(x)
        xs = [reassemble(xs[i]) for i, reassemble in enumerate(self.reassembles)]
        x = self.decoder(xs[::-1])
        x = self.head(x)
        return x.squeeze(dim=1)


class DPT_Depth(DPT):
    def __init__(self, config=config):
        features = config["decoder"]["features"]
        intermediate_features = config["head"]["intermediate_features"]
        head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(features, features // 2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, intermediate_features, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(intermediate_features, 1, 1),
            nn.ReLU(),
        )
        super().__init__(config, head)


# %%
if __name__ == "__main__":
    model = DPT_Depth()
    x = torch.randn((2, 3, 384, 384))
    output = model(x)
    print(output.shape)
# %%
