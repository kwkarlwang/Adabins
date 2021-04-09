#%%
from dataloader import NYUDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from models.tr import Transformer_Reassemble
from utils import *


#%%
class Experiment(pl.LightningModule):
    def __init__(self, hparams, verbose=True):
        super().__init__()
        self.config = vars(hparams)
        self.hparams = hparams
        self.model = self.__select_model(self.config)
        self.loss = self.__select_loss(self.config)
        self.verbose = verbose
        self.metrics = self.__init_metrics()

    def __init_metrics(self) -> dict:
        raise NotImplementedError()

    def __select_dataloader(self, config: dict, dataset_type: str):
        name = config["data"]["dataset"].lower()
        batch_size = config["data"]["batch_size"]
        num_workers = config["data"]["num_workers"]
        use_transform = config["data"].get("transform", False)
        if name == "nyu":
            if dataset_type == "train":
                return DataLoader(
                    NYUDataset("train", use_transform=use_transform),
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            elif dataset_type in ("validation", "test"):
                return DataLoader(
                    NYUDataset("test"), batch_size=batch_size, num_workers=num_workers
                )
        else:
            raise ValueError(f"Unknown dataset {name}")

    def __select_loss(self, config: dict):
        task = config["task"]
        if task == "depth":
            return L1Loss()
        elif task == "segmentation":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task {task}")

    def __select_model(self, config: dict):
        name = config["model"]
        if name == "transformer reassemble":
            return Transformer_Reassemble()
        else:
            raise ValueError(f"Unknown model {name}")

    def __get_batch_data(self, output, targets):
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        loss = self.loss(img)
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        return self.__get_batch_data(output, targets)

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor(outputs))
        data = {f"val_{key}": metric.compute() for key, metric in self.metrics.items()}
        data["val_loss"] = loss
        for key, value in data.items():
            self.log(key, value, prog_bar=True, sync_dist=True)

        for metric in self.metrics:
            metric.reset()
        if self.verbose:
            print("-" * 80)
            print(f"EPOCH {self.current_epoch} VALIDATION RESULT")
            for key, value in data.items():
                print(f"\t{key}: {value}")
            print("-" * 80)

    def test_step(self, batch, batch_idx):
        img, depth, semseg = batch
        targets = (semseg, depth)
        output = self(img)
        return self.__get_batch_data(output, targets)

    def test_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor(outputs))
        data = {f"test_{key}": metric.compute() for key, metric in self.metrics.items()}
        data["test_loss"] = loss
        for key, value in data.items():
            self.log(key, value, prog_bar=True, sync_dist=True)

        for metric in self.metrics:
            metric.reset()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optim

    def train_dataloader(self):
        return self.__select_dataloader(self.config, "train")

    def val_dataloader(self):
        return self.__select_dataloader(self.config, "validation")

    def test_dataloader(self):
        return self.__select_dataloader(self.config, "test")


class DepthExperiment(Experiment):
    def __init_metrics(self):
        return {"rel_err": RelativeError(), "rms_err": RootMeanSquaredError()}

    def __get_batch_data(self, output, targets):
        loss = self.loss(output, targets)
        depth_out = output
        semseg, depth = targets
        # calculate depth data
        self.metrics["rel_err"](depth_out, depth)
        self.metrics["rms_err"](depth_out, depth)
        return loss
