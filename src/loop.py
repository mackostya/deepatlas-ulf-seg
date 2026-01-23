import torch
import lightning as L
import psutil, os, gc
import torch.nn.functional as F

from torch import nn
from lightning.pytorch import loggers as pl_loggers
from monai.losses.dice import DiceLoss

from src.loop_interface import ModelInterface
from src.medsam2.model import MedSam2VolumetricSegmentor, MedSam2VolumetricSegmentor3D
from src.vnet.vnet import VNet


class TrainingLoop(ModelInterface, L.LightningModule):
    def __init__(self, model_type, class_weights=None, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights = None
        if model_type == "medsam2":
            self.model = MedSam2VolumetricSegmentor()
        elif model_type == "medsam2_3d":
            self.model = MedSam2VolumetricSegmentor3D(decoder_type="3d")
        elif model_type == "medsam2_3d_atlas":
            self.model = MedSam2VolumetricSegmentor3D(decoder_type="3d_atlas")
        elif model_type == "vnet":
            self.model = VNet(classes=9, use_atlas=False)
        elif model_type == "vnet_atlas":
            self.model = VNet(classes=9, use_atlas=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types are 'medsam2' and 'vnet'.")
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(
            include_background=True,
            reduction="none",
            to_onehot_y=True,
            softmax=True,
            weight=class_weights,
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        if len(batch) == 3:
            volumes, labels, _ = batch
            atlas = None
        elif len(batch) == 4:
            volumes, labels, _, atlas = batch
        else:
            raise ValueError(f"Batch must contain 3 or 4 elements, got {len(batch)} instead.")
        out = self.model(volumes, atlas=atlas)
        loss_ce = self.ce(out, labels)
        loss_dice = self.dice(out, labels.unsqueeze(1).long()).mean()
        loss = loss_ce + loss_dice
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_ce", loss_ce, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_dice", loss_dice, on_step=False, on_epoch=True, sync_dist=True)
        # if self.current_epoch % 5 == 0 and batch_idx == 1 and self.tb_logger is not None:
        #     self.log_tb_histograms()
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # validation_step defines the val loop.
        if len(batch) == 3:
            volumes, labels, _ = batch
            atlas = None
        elif len(batch) == 4:
            volumes, labels, _, atlas = batch
        else:
            raise ValueError(f"Batch must contain 3 or 4 elements, got {len(batch)} instead.")
        out = self.model(volumes, atlas=atlas)
        loss_ce = self.ce(out, labels)
        loss_dice = self.dice(out, labels.unsqueeze(1).long()).mean()
        loss = loss_ce + loss_dice

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss_ce", loss_ce, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss_dice", loss_dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "cpu_memory_usage",
            psutil.Process(os.getpid()).memory_info().rss / 1e9,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self.current_epoch % 20 == 0 and batch_idx == 0 and self.tb_logger is not None:
            slice_idx = volumes.shape[2] // 2  # middle slice
            labels = F.one_hot(labels.long().cpu(), num_classes=self.num_classes).movedim(-1, 1)
            img = volumes[0, :, slice_idx, :, :].detach().cpu().numpy()
            label = labels[0, :, slice_idx, :, :].detach().cpu().numpy()
            pred = out[0, :, slice_idx, :, :].detach().cpu().numpy()

            self.log_images(img, label, pred, seg_type="hipp")
            self.log_images(img, label, pred, seg_type="basal")
            self.log_images(img, label, pred, seg_type="extra")
            del img, label, pred, labels
            gc.collect()
            self.tb_logger.flush()
        return loss

    def on_fit_start(self):
        if self.tb_logger is None:
            # Get tensorboard logger
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break

            if tb_logger is None:
                print("No TensorBoard logger found. Images will not be logged.")
            self.tb_logger = tb_logger

    def configure_optimizers(self):
        all_params = []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue  # skip all frozen tensors
            all_params.append(p)
        optim_groups = [
            {"params": all_params, "lr": 1e-3},
        ]
        optimizer = torch.optim.Adam(optim_groups)
        return optimizer

    def forward(self, x, atlas=None):
        if atlas is not None:
            return self.model(x, atlas)
        else:
            return self.model(x)
