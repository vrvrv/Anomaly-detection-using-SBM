import copy
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.utils import make_grid

import src.models.utils as mutils


class SCORE_SDE(pl.LightningModule):
    def __init__(
            self,
            score_configs: dict,
            sde_configs: dict,
            sampler_configs: dict,
            lr: float,
            weight_decay: float,
            img_size: int,
            log_image_period: int,
            **kwargs
    ):
        super(SCORE_SDE, self).__init__()

        self.save_hyperparameters()

        self.score = mutils.get_model(
            **score_configs
        )
        self.ema_score = copy.deepcopy(self.score)

        self.sde, sampling_eps = mutils.get_sde(
            **sde_configs
        )

        self.sampler = mutils.get_sampling_fn(
            sde=self.sde,
            shape=(36, 3, img_size, img_size),
            sampling_eps=sampling_eps,
            **sampler_configs
        )

        self.register_buffer(
            "noise", torch.randn((36, 3, img_size, img_size))
        )

    def denoise(self, noise):
        return self.sampler(
            model=self.score, noise=noise
        )

    def shared_step(self, batch):
        t = torch.rand(len(batch), device=self.device) * (self.sde.T - 1e-5) + 1e-5
        z = torch.randn_like(batch)

        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z

        score = self.score(perturbed_data, t)

        # if not likelihood weighting
        loss = torch.square(score * std[:, None, None, None] + z)
        loss = torch.mean(loss.reshape(len(loss), -1), dim=-1)

        return loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X).mean()

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.current_epoch + 1 % self.hparams.log_image_period == 0:
            outs = self.denoise(self.noise)

            self.logger.log_image(
                key="Generated images", images=[make_grid(outs, nrow=6, normalize=True)]
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.9999),
            eps=1e-8
        )
        return optimizer
