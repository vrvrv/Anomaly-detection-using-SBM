import copy
import torch
import pytorch_lightning as pl

import src.models.utils as mutils
from src.models.SBM import VPSDE, VESDE, subVPSDE


class Score_SDE(pl.LightningModule):
    def __init__(self, sde: str, configs, sde_configs):
        super(Score_SDE).__init__()

        self.save_hyperparameters()

        self.score = mutils.get_model(
            configs
        )
        self.ema_score = copy.deepcopy(self.score)

        self.sde = eval(sde)(
            sde_configs
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
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        pass
