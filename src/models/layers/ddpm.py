import torch
import torch.nn as nn
import functools

from . import utils, blocks, normalization
from typing import Sequence

RefineBlock = blocks.RefineBlock
ResidualBlock = blocks.ResidualBlock
ResnetBlockDDPM = blocks.ResnetBlockDDPM
Upsample = blocks.Upsample
Downsample = blocks.Downsample
conv3x3 = blocks.ddpm_conv3x3
get_act = blocks.get_act
get_normalization = normalization.get_normalization
default_initializer = blocks.default_init


class DDPM(nn.Module):
    def __init__(
            self,
            sigma_0: float,
            sigma_1: float,
            num_scales: int = 1000,
            beta_0: float = 0.1,
            beta_1: float = 20.,
            dropout: float = 0.1,
            scale_by_sigma: bool = False,
            ema_rate: float = 0.9999,
            normalization: str = 'GroupNorm',
            nonlinearity: str = 'swish',
            nf: int = 128,
            ch_mult: Sequence[int] = (1, 2, 2, 2),
            num_res_blocks: int = 2,
            attn_resolutions: Sequence[int] = (16,),
            resamp_with_conv: bool = True,
            conditional: bool = True,
            fir: bool = False,
            fir_kernel: Sequence[int] = (1, 3, 3, 1),
            skip_rescale: bool = True,
            resblock_type: str = 'biggan',
            progressive: str = 'none',
            progressive_input: str = 'none',
            progressive_combine: str = 'sum',
            attention_type: str = 'ddpm',
            embedding_type: str = 'positional',
            init_scale: float = 0.,
            fourier_scale: int = 16,
            conv_size: int = 3,
            **kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)
        self.register_buffer(
            'sigmas', torch.tensor(utils.get_sigmas(sigma_0=sigma_0, sigma_1=sigma_1, num_scales=num_scales))
        )

        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = [config.data.image_size // (2 ** i) for i in range(self.num_resolutions)]

        AttnBlock = functools.partial(blocks.AttnBlock)
        self.conditional = conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        # Downsampling block
        modules.append(conv3x3(3, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, 3, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = scale_by_sigma

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = blocks.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # Input is in [0, 1]
        h = 2 * x - 1.

        # Downsampling block
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)

        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas

        return h
