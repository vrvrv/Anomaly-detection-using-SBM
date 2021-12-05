import copy
import wandb
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchmetrics import AUROC

import src.models.utils as mutils
from src.likelihood import get_likelihood_fn


def precision(score, label, topk=1, pos_label=0):
    _, sorted_idx = torch.sort(score)
    return (label == pos_label)[sorted_idx[-topk:]].float().mean()


def recall(score, label, topk=1, pos_label=0):
    _, sorted_idx = torch.sort(score)
    return (label[sorted_idx[-topk:]] == pos_label).float().sum() / (label == pos_label).float().sum()


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
        self.ema_model = copy.deepcopy(self.model).requires_grad_(False)

        self.sde, self.sampling_eps = mutils.get_sde(
            **sde_configs
        )

        self.sampler = mutils.get_sampling_fn(
            sde=self.sde,
            shape=(36, 3, img_size, img_size),
            sampling_eps=self.sampling_eps,
            **sampler_configs
        )

    def perturb_data(self, batch, t):
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        return perturbed_data, std, z

    def denoise(self, batch, t):

        sampler = mutils.get_sampling_fn(
            sde=self.sde,
            shape=(len(batch), 3, self.hparams.img_size, self.hparams.img_size),
            sampling_eps=self.sampling_eps,
            **self.hparams.sampler_configs
        )

        vec_t = torch.ones(batch.shape[0], device=batch.device) * t
        perturbed_data, _, _ = self.perturb_data(batch, vec_t)
        return sampler(
            model=self.model, device=self.device, init=perturbed_data, t=t
        )

    def shared_step(self, batch):
        t = torch.rand(len(batch), device=self.device) * (self.sde.T - 1e-5) + 1e-5
        perturbed_data, std, z = self.perturb_data(batch, t)

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
        self.likelihood_fn = get_likelihood_fn(sde=self.sde)

    def test_step(self, batch, batch_idx):
        X, mask, y = batch

        X_recon = self.denoise(X, t=0.15)

        if self.hparams.on_white_background:
            white_background = (X > 0.8).float()
            X_recon = X * white_background + (1. - white_background) * X_recon

        if self.hparams.on_black_background:
            black_background = (X < -0.8).float()
            X_recon = X * black_background + (1. - black_background) * X_recon

        err = torch.square(X - X_recon).mean(dim=1, keepdim=True)
        shape = err.shape
        err_flatten = err.reshape(shape[0], -1)

        mask = (err_flatten <= err_flatten.quantile(self.hparams.err_quantile, dim=1, keepdim=True)).reshape(
            shape).float()

        self.logger.log_image(
            "Reconstruction",
            images=[torch.cat([x, x_recon, m.repeat(3, 1, 1)], dim=1) for x, x_recon, m in zip(X, X_recon, mask)]
        )

        bpd = self.likelihood_fn(model=self.model, data=X)
        c_bpd = self.likelihood_fn(
            model=self.model,
            data=X,
            mask=mask
        )

        err = err.mean(dim=(1, 2, 3))

        return X, bpd, c_bpd, err, y

    def test_epoch_end(self, outputs) -> None:
        images = []
        bpd = []
        c_bpd = []
        err = []
        label = []

        for X, bpd_i, c_bpd_i, err_i, y_i in outputs:
            images.append(X)
            bpd.append(bpd_i)
            c_bpd.append(c_bpd_i)
            err.append(err_i)
            label.append(y_i)

        images = torch.cat(images)
        bpd = torch.cat(bpd)
        c_bpd = torch.cat(c_bpd)
        err = torch.cat(err)
        label = torch.cat(label)

        # anomalous / non-anomalous data label : 1 / 0
        # bpd and c_bpd is likelihood -> larger for non-anomalous data
        # err is error -> smaller for non-anomalous data
        auroc = AUROC(num_classes=2, pos_label=0)

        self.log("AUROC_unconditioned", auroc(bpd, label), logger=True)
        self.log("AUROC_conditioned", auroc(c_bpd, label), logger=True)
        self.log("AUROC_recon_err", auroc(-err, label), logger=True)

        # self.log("P@5_unconditioned", precision(bpd, label, topk=5, pos_label=0), logger=True)
        # self.log("R@5_unconditioned", recall(bpd, label, topk=5, pos_label=0), logger=True)
        #
        # self.log("P@5_conditioned", precision(c_bpd, label, topk=5, pos_label=0), logger=True)
        # self.log("R@5_conditioned", recall(c_bpd, label, topk=5, pos_label=0), logger=True)



        # bpd and c_bpd is likelihood -> larger for non-anomalous data
        auroc = AUROC(num_classes=2, pos_label=0)

        self.log("AUROC_unconditioned", auroc(bpd, label), logger=True)
        self.log("AUROC_conditioned", auroc(c_bpd, label), logger=True)

        self.log("P@5_unconditioned", precision(bpd, label, topk=5, pos_label=0), logger=True)
        self.log("R@5_unconditioned", recall(bpd, label, topk=5, pos_label=0), logger=True)

        self.log("P@5_conditioned", precision(c_bpd, label, topk=5, pos_label=0), logger=True)
        self.log("R@5_conditioned", recall(c_bpd, label, topk=5, pos_label=0), logger=True)

        self.logger.log_table(
            key="result",
            columns=["image", "full_loglikelihood", "cond_loglikelihood", "label"],
            data=[[wandb.Image(x), ll, cll, y] for x, ll, cll, y in zip(images, bpd, c_bpd, label)]
        )
        # self.log("AUROC_conditioned_delta=0.5", auroc(c_bpd_05, y), logger=True)
        # self.log("AUROC_conditioned_delta=0.8", auroc(c_bpd_08, y), logger=True)

    @pl.utilities.rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.hparams.log_image_period == 0:
            outs = self.sampler(
                model=self.model, device=self.device
            )

            self.logger.log_image(
                key="Generated images", images=[make_grid(outs, nrow=6, normalize=True)]
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.9999), eps=1e-8
        )
        return optimizer
