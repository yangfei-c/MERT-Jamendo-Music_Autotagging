import logging
import os
#=============屏蔽警告以及其它打印的无用信息========================
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")
warnings.filterwarnings("ignore", message="No positive samples in targets")
warnings.filterwarnings("ignore", ".*Consider setting `persistent_workers=True`.*")
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import hydra
import pytorch_lightning as pl
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
#===========================================================================================
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('medium')
#
from data_loader import MTGJamendoDataModule
from trainer.linear_trainer import MTGJamendoTrainer as LinearTrainer
from trainer.gcn_trainer import MTGJamendoTrainer as GCNTrainer


def setup_callbacks(cfg: DictConfig):
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if 'early_stopping' in cfg.training:
        es = cfg.training.early_stopping
        callbacks.append(EarlyStopping(
            monitor=es.monitor,
            patience=es.patience,
            mode=es.mode,
            verbose=False
        ))

    ckpt_dir = Path(cfg.paths.checkpoint_dir) / cfg.project.name
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    ))
    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.project.seed)

    print("=" * 60)
    # 打印当前的配置信息
    print(f"🚀 Training Project: {cfg.project.name}")
    print(f"📌 Task: {cfg.task_name} | Model: {cfg.model_type}")
    print("=" * 60)

    print(">>> Initializing Data Module...")
    data_module = MTGJamendoDataModule(cfg)
    data_module.setup()

    # 获取当前任务的类别数 (例如 genre 可能是 87 类，instrument 可能是 40 类)
    num_classes = data_module.train_dataset.num_classes
    print(f"ℹ️ Detected {num_classes} classes for task '{cfg.task_name}'")

    #==========动态选择 Trainer==========
    if cfg.model_type.lower() == "linear":
        model_system = LinearTrainer(cfg, num_classes)
    elif cfg.model_type.lower() == "gcn":
        model_system = GCNTrainer(cfg, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}. Supported: [linear, gcn]")
    # =================================

    logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name=cfg.project.name, version='')
    callbacks = setup_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_model_summary=False
    )

    print("\n>>> Start Training...")
    trainer.fit(model_system, datamodule=data_module)

    print("\n>>> Start Testing (Loading Best Model)...")
    trainer.test(model_system, datamodule=data_module, ckpt_path='best')
    print("\n✅ All Done!")


if __name__ == "__main__":
    main()