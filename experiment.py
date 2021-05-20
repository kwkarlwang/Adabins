from dataloader import NYUDataset
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader  # For custom data-sets
import pytorch_lightning as pl
from models.unet_adaptive_bins import UnetAdaptiveBins
from utils import L1Loss, RelativeError, RootMeanSquaredError, EigenLoss
from typing import Union, Tuple
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt
import os

# %%
class Experiment(pl.LightningModule):
    def __init__(self, config, verbose=True):
        super().__init__()
        # this will save the hyperparameters
        self.config = config
        self.verbose = verbose
        self.model = self._select_model(self.config["model"])
        self.loss = self._select_loss(self.config["experiment"])
        experiment_folder = "experiment_data"
        self.path = f"./{experiment_folder}/{config['name']}/{config['version']}"
        self.metrics = self._init_metrics()
        self.save_results = config.get("save_results", [])

    def _init_metrics(self) -> dict:
        raise NotImplementedError()

    def _select_dataloader(self, config: dict, dataset_type: str):
        name = config["name"].lower()
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        use_transform = config.get("use_transform", False)
        if name == "nyu":
            if dataset_type == "train":
                return DataLoader(
                    NYUDataset("train", use_transform=use_transform),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                )
            elif dataset_type in ("val", "test"):
                return DataLoader(
                    NYUDataset(dataset_type),
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
        else:
            raise ValueError(f"Unknown dataset {name}")

    def _select_model(self, config: dict):
        raise NotImplementedError()

    def _select_loss(self, config: dict):
        raise NotImplementedError()

    def _get_batch_data(
        self, output: Union[Tensor, Tuple[Tensor]], target: Tuple[Tensor, Tensor]
    ):
        raise NotImplementedError()

    def _select_output_target(
        self, output: Union[Tensor, Tuple[Tensor]], target: Tuple[Tensor, Tensor]
    ):
        return output, target

    def _post_processing(self, output: Tensor):
        return output

    def _output_to_image(self, idx, img, output, target):
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        img, depth, semseg = batch
        target = (semseg, depth)
        output = self(img)
        output, target = self._select_output_target(output, target)
        output = self._post_processing(output)
        loss = self.loss(output, target)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        if self.verbose and batch_idx % 20 == 0:
            print(f"EPOCH {self.current_epoch}\titer:{batch_idx}\tloss: {loss}")
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        img, depth, semseg = batch
        target = (semseg, depth)
        output = self(img)
        return self._get_batch_data(output, target)

    def validation_epoch_end(self, output: list):
        loss = torch.mean(torch.tensor(output))
        data = {f"val_{key}": metric.compute() for key, metric in self.metrics.items()}
        data["val_loss"] = loss
        for key, value in data.items():
            self.log(key, value, prog_bar=True, sync_dist=True)

        for metric in self.metrics.values():
            metric.reset()
        if self.verbose:
            print("-" * 80)
            print(f"EPOCH {self.current_epoch} VALIDATION RESULT")
            for key, value in data.items():
                print(f"\t{key}: {value}")
            print("-" * 80)

    def test_step(self, batch: Tensor, batch_idx: int):
        img, depth, semseg, orig_img = batch
        target = (semseg, depth)
        output = self(img)
        if batch_idx in self.save_results:
            self._output_to_image(batch_idx, orig_img, output, target)
        return self._get_batch_data(output, target)

    def test_epoch_end(self, output: list):
        loss = torch.mean(torch.tensor(output))
        data = {f"test_{key}": metric.compute() for key, metric in self.metrics.items()}
        data["test_loss"] = loss
        for key, value in data.items():
            self.log(key, value, prog_bar=True, sync_dist=True)

        for metric in self.metrics.values():
            metric.reset()

    # MUST OVERRIDE
    def configure_optimizers(self):
        lr = self.config["experiment"]["learning_rate"]
        if self.config["model"]["name"] == "adabins":
            params = [
                {"params": self.model.get_1x_lr_params(), "lr": lr / 10},
                {"params": self.model.get_10x_lr_params(), "lr": lr},
            ]
        else:
            params = self.parameters()

        optim = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.2, patience=5, verbose=True
        )
        # return [optim], [scheduler]
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": self.monitor}

    def train_dataloader(self):
        return self._select_dataloader(self.config["dataset"], "train")

    def val_dataloader(self):
        return self._select_dataloader(self.config["dataset"], "val")

    def test_dataloader(self):
        return self._select_dataloader(self.config["dataset"], "test")


class DepthExperiment(Experiment):
    def __init__(self, config, verbose=True):
        super().__init__(config, verbose=verbose)
        self.monitor = "val_rms_err"

    def _select_model(self, config: dict):
        name = config.get("name", "adabins").lower()
        n_bins = config.get("n_bins", 256)
        min_depth = config.get("min_depth", 1e-3)
        max_depth = config.get("min_depth", 10)
        norm = config.get("norm", "linear")
        if name == "adabins":
            # TODO: pass in the config
            model = UnetAdaptiveBins.build(
                n_bins=n_bins, min_val=min_depth, max_val=max_depth, norm=norm
            )
            return model
        else:
            raise ValueError(f"Unknown model {name}")

    def _init_metrics(self):
        device = (
            "cuda"
            if self.config.get("gpus", 1) > 0 and torch.cuda.is_available()
            else "cpu"
        )
        return {
            "rel_err": RelativeError(device=device),
            "rms_err": RootMeanSquaredError(device=device),
        }

    def _get_batch_data(self, output, target):
        output, target = self._select_output_target(output, target)
        output = self._post_processing(output)
        loss = self.loss(output, target)
        print(output.shape, target.shape)
        # calculate depth data
        self.metrics["rel_err"](output, target)
        self.metrics["rms_err"](output, target)
        return loss

    def _select_output_target(self, output: Tensor, target: Tuple[Tensor, Tensor]):
        return output, target[1]

    def _select_loss(self, config: dict):
        loss = config["loss"].lower()
        if loss == "l1":
            return L1Loss()
        elif loss == "l2":
            return nn.MSELoss()
        elif loss == "eigen":
            lamb = config.get("lambda", 0.5)
            return EigenLoss(lamb)
        else:
            raise ValueError(f"Unknown loss {loss}")

    def _post_processing(self, output: Tensor):
        output = tf.resize(output, self.config["dataset"]["img_size"])
        # back to 480x640
        return output

    def _output_to_image(self, idx, img, output, target):
        output, target = self._select_output_target(output, target)
        output = self._post_processing(output)
        path = f"{self.path}/test_{idx}"
        os.makedirs(path, exist_ok=True)
        for i in range(output.shape[0]):
            im = Image.fromarray((img[i].cpu().numpy()))
            im.save(f"{path}/img_{i}.png")

            normalize_term = max(
                target[i].max().cpu().item(), output[i].max().cpu().item()
            )

            plt.figure(figsize=(8, 6))
            plt.imshow(output[i].cpu() / normalize_term)
            plt.axis("off")
            plt.savefig(f"{path}/depth_{i}", bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.imshow(target[i].cpu() / normalize_term)
            plt.axis("off")
            plt.savefig(f"{path}/depth_{i}_gt", bbox_inches="tight")
            plt.close()
