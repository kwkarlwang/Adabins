# %%
import pytorch_lightning as pl
from file_utils import read_file, dict_to_args, write_to_file
import sys
from experiment import DepthExperiment, Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def get_experiment(task: str) -> Experiment:
    if task == "depth":
        return DepthExperiment
    else:
        raise ValueError(f"Unknown task {task}")


# %%
if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    resume_from_checkpoint = bool(sys.argv[2] == "1") if len(sys.argv) > 2 else None
    fast_dev_run = int(sys.argv[3]) if len(sys.argv) > 3 else False

    config = read_file(exp_name)
    hparams = dict_to_args(config)

    experiment_folder = "experiment_data"
    checkpoint_path = (
        config["experiment"]["checkpoint_path"]
        if "checkpoint_path" in config["experiment"]
        else False
    )
    if resume_from_checkpoint and not checkpoint_path:
        path = f"./{experiment_folder}/{config['name']}/{config['version']}/checkpoints"
        try:
            checkpoint = "last.ckpt"
            checkpoint_path = os.path.join(path, checkpoint)
            print(f"RESUMING FROM {checkpoint_path}")
        except Exception:
            print("FAILED TO LOAD")

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=f"./{experiment_folder}",
        name=str(config["name"]),
        version=str(config["version"]),
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU"
        if config["experiment"]["task"] != "depth"
        else "val_rel_err",
        mode="max" if config["experiment"]["task"] != "depth" else "min",
        save_last=True,
    )

    print("-" * 80)
    print("RUNNING FOLLOWING EXPERIMENT")
    for key, value in config.items():
        print(f"\t{key}: {value}")
    print("-" * 80)

    experiment = get_experiment(config["experiment"]["task"])
    if resume_from_checkpoint or checkpoint_path:
        exp = experiment.load_from_checkpoint(
            checkpoint_path, hparams=hparams, config=config
        )
    else:
        exp = experiment(hparams, config=config)

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        resume_from_checkpoint=checkpoint_path,
        gpus=config["gpus"],
        max_epochs=config["experiment"]["epochs"],
        progress_bar_refresh_rate=False,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(exp)
    if not fast_dev_run:
        trainer.test()