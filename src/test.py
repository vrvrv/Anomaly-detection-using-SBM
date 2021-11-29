from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.loggers.wandb import WandbLogger

from src import utils


log = utils.get_logger(__name__)


def test(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random

    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        img_size=config.datamodule.img_size
    )

    # Init lightning loggers
    log.info(f"Instantiating logger <{config.logger._target_}>")
    logger = hydra.utils.instantiate(config.logger)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, _convert_="partial"
    )

    log.info("Start testing")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.trainer.resume_from_checkpoint)

    # Make sure everything closed properly
    if isinstance(logger, WandbLogger):
        import wandb

        wandb.finish()