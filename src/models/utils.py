from typing import Sequence
from src.models.SBM.layers import (
    DDPM,
    NCSNv2,
    NCSNpp
)
from src.models.SBM.sde import (
    VPSDE,
    subVPSDE,
    VESDE
)

model_dict = {
    'ddpm': DDPM,
    'ncsnv2': NCSNv2,
    'ncsnpp': NCSNpp
}

sde_dict = {
    'vpsde': DDPM,
    'subvpsde': NCSNv2,
    'vesde': NCSNpp
}


def get_model(
        model_name: str,
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
    model = model_dict[model_name.lower()](
        sigma_0=sigma_0,
        sigma_1=sigma_1,
        num_scales=num_scales,
        beta_0=beta_0,
        beta_1=beta_1,
        dropout=dropout,
        scale_by_sigma=scale_by_sigma,
        ema_rate=ema_rate,
        normalization=normalization,
        nonlinearity=nonlinearity,
        nf=nf,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        resamp_with_conv=resamp_with_conv,
        conditional=conditional,
        fir=fir,
        fir_kernel=fir_kernel,
        skip_rescale=skip_rescale,
        resblock_type=resblock_type,
        progressive=progressive,
        progressive_input=progressive_input,
        progressive_combine=progressive_combine,
        attention_type=attention_type,
        embedding_type=embedding_type,
        init_scale=init_scale,
        fourier_scale=fourier_scale,
        conv_size=conv_size,
    )

    return model


def get_sde(
        sde_name: str,
        sigma_0: float,
        sigma_1: float,
        beta_0: float,
        beta_1: float,
        N: int

):
    sde = sde_dict[sde_name.lower()](
        sigma_0=sigma_0,
        sigma_1=sigma_1,
        beta_0=beta_0,
        beta_1=beta_1,
        N=N
    )

    return sde
