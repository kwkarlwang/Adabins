#%%
from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from utils import *
from file_utils import *
import sys
import time
from factory import select_dataloader
from experiment import *
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def get_experiment(task: str) -> Experiment:
    if task == "depth":
        return DepthExperiment
    else:
        raise ValueError(f"Unknown task {task}")


#%%
if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    resume_from_checkpoint = bool(sys.argv[2] == "1") if len(sys.argv) > 2 else None
    fast_dev_run = int(sys.argv[3]) if len(sys.argv) > 3 else False
    #%%
    config_dict = read_file(exp_name)
    hparams = dict_to_args(config_dict)
    #%%
    experiment_folder = (
        "experiment_nyu"
        if "nyu" in config_dict.get("dataset", "hypersim")
        else "experiment_data"
    )
    checkpoint_path = hparams.checkpoint_path if "checkpoint_path" in hparams else False
    if resume_from_checkpoint and not checkpoint_path:
        path = f"./{experiment_folder}/{config_dict['experiment_name']}/{config_dict['version_number']}/checkpoints"
        try:
            checkpoint = "last.ckpt"
            checkpoint_path = os.path.join(path, checkpoint)
            print(f"RESUMING FROM {checkpoint_path}")
        except:
            print("FAILED TO LOAD")

    #%%
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=f"./{experiment_folder}",
        name=f'{str(config_dict["experiment_name"])}',  # This will create different subfolders for your models
        version=f'{str(config_dict["version_number"])}',
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU" if hparams.model_type != "depth" else "val_rel_err",
        mode="max" if hparams.model_type != "depth" else "min",
        save_last=True,
    )
    #%%
    print("-" * 80)
    print("RUNNING FOLLOWING EXPERIMENT")
    for key, value in config_dict.items():
        print(f"\t{key}: {value}")
    print("-" * 80)
    #%%
    experiment = get_experiment(config_dict["task"])
    if resume_from_checkpoint or checkpoint_path:
        exp = experiment.load_from_checkpoint(checkpoint_path, hparams=hparams)
    else:
        exp = experiment(hparams)

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        resume_from_checkpoint=checkpoint_path,
        gpus=1,
        max_epochs=hparams.num_epochs,
        progress_bar_refresh_rate=False,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(exp)
    if not fast_dev_run:
        trainer.test()