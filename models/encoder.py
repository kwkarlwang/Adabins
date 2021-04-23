# %%
import timm
import torch
import torch.nn as nn
from typing import Callable, Dict


encoder_config = {"hooks": [2, 5, 8, 11], "dropout": 0.1, "name": "base"}


def save_output_hook(output_name: str, output_dict: dict) -> Callable:
    def fn(a, b, output: torch.Tensor) -> None:
        output_dict[output_name] = output

    return fn


class Encoder(nn.Module):
    def __init__(self, config: dict = encoder_config):

        super(Encoder, self).__init__()
        name = config["name"]
        if name == "hybrid":
            model_name = "vit_base_resnet50_384"
        elif name == "large":
            model_name = "vit_large_patch16_384"
        else:
            model_name = "vit_base_patch16_384"
        self.transformer = timm.create_model(
            model_name, pretrained=True, drop_rate=config["dropout"]
        )
        hooks = config["hooks"]
        modules = dict(self.transformer.named_modules())
        self.output = {}

        for hook in hooks:
            layer_name = f"blocks.{hook}"
            modules[layer_name].register_forward_hook(
                save_output_hook(hook, self.output)
            )

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        self.transformer(x)
        # output shape: [(batch_size, number of patch + 1, embedded feature size)]
        return [self.output[key] for key in sorted(self.output)]


if __name__ == "__main__":
    model = Encoder()
    # batch_size, channels, img_h, img_w
    x = torch.randn((2, 3, 384, 384))
    output = model(x)
    print(output[11].shape)
