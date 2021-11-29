import torch
import torch.nn as nn
from typing import Sequence
from itertools import chain
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchvision.utils import make_grid

from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class ENC(nn.Module):
    def __init__(self, shared_layer, latent_dim):
        super().__init__()

        self.layer = shared_layer
        self.fc_mu = nn.Linear(self.layer.output_dim, latent_dim)
        self.fc_logstd = nn.Linear(self.layer.output_dim, latent_dim)

    def forward(self, x):
        x = self.layer(x)

        mu = self.fc_mu(x)
        log_std = self.fc_logstd(x)

        return mu, log_std


class CNNEncoder(nn.Module):
    def __init__(
            self,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            strides: Sequence[int],
            hidden_dims: Sequence[int],
            **kwargs
    ):
        super().__init__()
        channel_chain = list(chain([3], channels))

        self.cnn = nn.ModuleList(
            [
                nn.Conv2d(
                    channel_chain[i], channel_chain[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i]
                ) for i in range(len(channel_chain) - 1)
            ]
        )

        self.bn = nn.ModuleList(
            [
                nn.BatchNorm2d(channel_chain[i]) for i in range(1, len(channel_chain))
            ]
        )

        self.dense = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dims[i], hidden_dims[i + 1]
                ) for i in range(len(hidden_dims) - 1)
            ]
        )
        self.act = nn.ReLU()
        self.flatten = Flatten()
        self._output_dim = hidden_dims[-1]

    def forward(self, x):
        for l, bn in zip(self.cnn, self.bn):
            x = self.act(bn(l(x)))

        x = self.flatten(x)

        for i, l in enumerate(self.dense):
            x = self.act(l(x))

        return x

    @property
    def output_dim(self):
        return self._output_dim


class CNNDecoder(nn.Module):
    def __init__(
            self,
            hidden_dims,
            channels,
            kernel_sizes,
            strides,
            **kwargs
    ):
        super().__init__()
        channel_chain = list(chain(channels, [3]))

        self.unflatten = UnFlatten()
        self.decnn = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channel_chain[i], channel_chain[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i]
                ) for i in range(len(channel_chain) - 1)
            ]
        )

        self.bn = nn.ModuleList(
            [
                nn.BatchNorm2d(channel_chain[i]) for i in range(1, len(channel_chain))
            ]
        )

        self.dense = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dims[i], hidden_dims[i + 1]
                ) for i in range(len(hidden_dims) - 1)
            ]
        )
        self.act = nn.ReLU()
        self.unflatten = UnFlatten()
        self._output_dim = hidden_dims[-1]

        self.tail = nn.Tanh()

    def forward(self, x):
        for i, l in enumerate(self.dense):
            x = self.act(l(x))

        x = self.unflatten(x)

        for i, (l, bn) in enumerate(zip(self.decnn, self.bn)):
            x = bn(l(x))

            if i != len(self.decnn) - 1:
                x = self.act(x)
        return self.tail(x)

    @property
    def output_dim(self):
        return self._output_dim


class AutoEncoder(pl.LightningModule):
    def __init__(
            self, encoder_conf, decoder_conf, lr, **kwargs
    ):
        super(AutoEncoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = CNNEncoder(**encoder_conf)
        self.decoder = CNNDecoder(**decoder_conf)

    def forward(self, x):
        return self.encoder(x)

    def recon_err(self, x):
        xhat = self.decoder(self.encoder(x))
        return torch.square(xhat - x).mean(dim=(1, 2, 3))

    def get_mask(self, x, delta):
        err = torch.abs(self.decoder(self.encoder(x)) - x).mean(dim=1, keepdim=True)

        shape = err.shape

        err_flatten = err.reshape(shape[0], -1)

        mask = torch.logical_or(
            (err_flatten < err_flatten.quantile(delta, dim=1, keepdim=True)).reshape(shape),
            x.mean(dim=1, keepdim=True) == 1
        ).float()

        return mask

    def shared_step(self, X):
        Xhat = self.decoder(self.encoder(X))
        return F.mse_loss(X, Xhat)

    def training_step(self, batch, batch_idx):
        X, y = batch

        loss = self.shared_step(X)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        loss = self.shared_step(X)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % self.hparams.log_image_period == 0:
            image_pair = torch.cat([X[:8], self.decoder(self.encoder(X[:8]))], dim=0)

            self.logger.log_image(
                key="Reconstruction",
                images=[make_grid(image_pair, ncol=2, normalize=True)]
            )

    def test_step(self, batch, batch_idx):
        X, y = batch

        recon_err = self.recon_err(X)

        return recon_err, y

    def test_epoch_end(self, outputs) -> None:
        recon_err = []
        y = []

        for recon_err_i, y_i in outputs:
            recon_err.append(recon_err_i)
            y.append(y_i)

        recon_err = torch.cat(recon_err)
        y = torch.cat(y)

        auroc = AUROC(num_classes=2, pos_label=1)

        self.log("AUROC", auroc(recon_err, y), logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr
        )
