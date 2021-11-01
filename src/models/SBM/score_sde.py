import copy
import torch
import torch.optim as optim
import pytorch_lightning as pl

import src.models.utils as mutils
from src.models.SBM import *


class SCORE_SDE(pl.LightningModule):
    def __init__(
            self,
            score_configs: dict,
            sde_configs: dict,
            lr: float,
            weight_decay: float
    ):
        super(SCORE_SDE).__init__()

        self.save_hyperparameters()

        self.score = mutils.get_model(
            **score_configs
        )
        self.ema_score = copy.deepcopy(self.score)

        self.sde = mutils.get_sde(
            **sde_configs
        )

    def shared_step(self, batch):
        t = torch.rand(len(batch), device=self.device)
        z = torch.randn_like(batch)

        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std * z
        score = self.score(perturbed_data, t)

        # if not likelihood weighting
        loss = torch.square(score * std + z)
        loss = torch.mean(loss.reshape(len(loss), -1), dim=-1)

        return loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.9999),
            eps=1e-8
        )
        return optimizer
