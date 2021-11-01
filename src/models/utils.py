from typing import List, Sequence
# from src.models.SBM.layers import (
#     DDPM,
#     NCSNv2,
#     NCSNpp
# )
from src.models.SBM.sde import (
    VPSDE,
    subVPSDE,
    VESDE
)
from src.models.blocks import (
    RefineNet,
    WideResnet
)

model_dict = {
    'ddpm': WideResnet,
    'ncsn': RefineNet
}

sde_dict = {
    'vpsde': VPSDE,
    'subvpsde': subVPSDE,
    'vesde': VESDE
}


def get_model(
        model_name: str,
        T: int,
        ch: int,
        ch_mult: List[int],
        attn: List[int],
        num_res_blocks: int,
        dropout: float,
        **kwargs
):
    model = model_dict[model_name.lower()](
        ## DDPM ##
        T=T,
        ch=ch,
        ch_mult=ch_mult,
        attn=attn,
        num_res_blocks=num_res_blocks,
        dropout=dropout
    )

    return model


def get_sde(
        sde_name: str,
        sigma_0: float = 0.01,
        sigma_1: float = 50.,
        beta_0: float = 0.1,
        beta_1: float = 20.,
        N: int = 1000,
        **kwargs
):
    sde = sde_dict[sde_name.lower()](
        sigma_0=sigma_0,
        sigma_1=sigma_1,
        beta_0=beta_0,
        beta_1=beta_1,
        N=N
    )

    return sde
