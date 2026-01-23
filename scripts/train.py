import torch
import lightning as L
import torch.utils.data as data
import torchio as tio

from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from src import utils
from src.loop import TrainingLoop
from dataloaders.dataset import LISADataset

if __name__ == "__main__":
    config_id = 0
    log_dir, cfg = utils.init_system(model_type="medsam2", config_id=config_id)
    torch.set_float32_matmul_precision("medium")

    batch_size = cfg["batch_size"]
    try:
        num_epochs = cfg["num_epochs"] if cfg["num_epochs"] else 1000
    except KeyError:
        num_epochs = 1000
    use_atlas = "atlas" in cfg["model_type"]

    transfroms = [
        tio.RandomElasticDeformation(),
        tio.RandomAffine(),
        tio.RandomAnisotropy(),
        tio.RandomGhosting(),
        tio.RandomSpike(),
        tio.RandomNoise(),
    ]
    train_dataset = LISADataset(
        patch_size=cfg["patch_size"],
        split="train",
        transforms=transfroms,
        use_atlas=use_atlas,
        interpolate=cfg["interpolate"],
    )
    val_dataset = LISADataset(
        patch_size=cfg["patch_size"],
        split="val",
        transforms=None,
        use_atlas=use_atlas,
        interpolate=cfg["interpolate"],
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name="tb_logger")
    # wandb_logger = pl_loggers.WandbLogger(
    #     save_dir=log_dir,
    #     name="wandb_logger",
    #     project="LISA25",
    # )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=5,
        persistent_workers=False,
        # pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=5,
        persistent_workers=False,
        # pin_memory=True,
    )

    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # train the model
    loop = TrainingLoop(model_type=cfg["model_type"], class_weights=cfg["class_weights"])
    trainer = L.Trainer(
        logger=tb_logger,  # or use your wandb_logger
        max_epochs=num_epochs,
        callbacks=[checkpoint],
        devices=1,
        accelerator="gpu",
        log_every_n_steps=5,
    )
    trainer.fit(model=loop, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
