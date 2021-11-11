from typing import List, Sequence
from src.models.SBM.sde import (
    VPSDE,
    subVPSDE,
    VESDE
)
from src.models.blocks import (
    RefineNet,
    WideResnet
)
from src.models.SBM.sampling import get_sampling_fn

model_dict = {
    'ddpm': WideResnet,
    'ncsn': RefineNet
}

sde_dict = {
    'vpsde': VPSDE,
    'subvpsde': subVPSDE,
    'vesde': VESDE
}

sampling_eps_dict = {
    'vpsde': 1e-3,
    'subvpsde': 1e-3,
    'vesde': 1e-5
}


def get_model(
        model_name: str,
        ch: int,
        ch_mult: List[int],
        attn: List[int],
        num_res_blocks: int,
        dropout: float,
        **kwargs
):
    model = model_dict[model_name.lower()](
        ## DDPM ##
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

    sampling_eps = sampling_eps_dict[sde_name.lower()]

    return sde, sampling_eps


def get_sampling_fn(
        predictor_name: str,
        corrector_name: str,
        sde,
        shape,
        sampling_eps: float,
        snr,
        n_steps,
        probability_flow,
        continuous,
        denoise,
        device,
        **kwargs
):
    sampler = get_sampling_fn(
        predictor_name=predictor_name,
        corrector_name=corrector_name,
        sde=sde,
        shape=shape,
        eps=sampling_eps,
        snr=snr,
        n_steps=n_steps,
        continuous=continuous,
        denoise=denoise,
        device=device,
    )
    return sampler
