import os
import sys
import random
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.dataset import MRITextDataModule
from models.fine_grain_cross_attention_textquery import FineGrainedMRITextCrossAttentionModel as FineGrainedMRITextCrossAttentionModelTextQuery
from configs.config import config

seed = config.seed if config.seed is not None else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
pl.seed_everything(seed, workers=True)


def main():
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    ## spin up data loaders and datasets
    datamodule = MRITextDataModule(
        train_csv=config.train_csv,
        val_csv=config.val_csv,
        image_dir=config.image_dir,
        text_dir=config.text_dir,
        cfg=config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    ## load cross attention model 
    if config.model_type == "cross_attention_textquery":
        model = FineGrainedMRITextCrossAttentionModelTextQuery()
        freeze_backbone = True
        if freeze_backbone:
            for param in model.vit.backbone.parameters():
                param.requires_grad = False
        monitor_metric = "val_auc"
        filename_template = "textquery_crossattention_{epoch:02d}-{val_f1:.4f}-{val_auc:.4f}"
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # checkpoint callback 
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        monitor=monitor_metric,
        mode="max",
        save_top_k=20,
        save_last=False,
        filename=filename_template
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    ## logger 
    logger = None
    if config.use_wandb:
        logger = WandbLogger(
            project=config.project_name,
            name=config.run_name,
            log_model=True,
            save_dir=config.checkpoint_dir
        )

    # start trainer 
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices=[1],
        precision="16-mixed",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
