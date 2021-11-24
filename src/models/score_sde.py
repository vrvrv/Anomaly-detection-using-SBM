import copy
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchmetrics import AUROC, PrecisionRecallCurve

from src.likelihood import get_likelihood_fn
import src.models.utils as mutils

from src.models.sde import (
    VPSDE,
    subVPSDE,
    VESDE
)


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

        self.model = mutils.get_model(
            **score_configs
        )
        self.ema_model = copy.deepcopy(self.model)

        self.sde, sampling_eps = mutils.get_sde(
            **sde_configs
        )

        self.sampler = mutils.get_sampling_fn(
            sde=self.sde,
            shape=(36, 3, img_size, img_size),
            sampling_eps=sampling_eps,
            **sampler_configs
        )

    def denoise(self):
        return self.sampler(
            model=self.model, device=self.device
        )

    def shared_step(self, batch):
        t = torch.rand(len(batch), device=self.device) * (self.sde.T - 1e-5) + 1e-5
        z = torch.randn_like(batch)

        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z

        score_fn = mutils.get_score_fn(self.sde, self.model, train=True)
        score = score_fn(perturbed_data, t)

        loss = torch.mean(
            torch.square(score * std[:, None, None, None] + z)
        )
        return loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.shared_step(X)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_test_start(self) -> None:
        self.likelihood = get_likelihood_fn(sde=self.sde)

    def test_step(self, batch, batch_idx):
        X, y = batch

        bpd = self.likelihood(model=self.model, data=X)

        return bpd, y

    def test_epoch_end(self, outputs) -> None:
        bpd = []
        y = []

        for bpd_i, y_i in outputs:
            bpd.append(bpd_i)
            y.append(y_i)

        bpd = torch.cat(bpd)
        y = torch.cat(y)

        auroc = AUROC(num_classes=2, pos_label=0)

        self.log("AUROC", auroc(bpd, y), logger=True)

    @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.hparams.log_image_period == 0:
            outs = self.denoise()
            self.logger.log_image(
                key="Generated images", images=[make_grid(outs, nrow=6, normalize=True)]
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.9999), eps=1e-8
        )
        return optimizer
