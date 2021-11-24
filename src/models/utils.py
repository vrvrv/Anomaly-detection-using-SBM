import torch

from typing import List
from src.models.sde import (
    VPSDE,
    subVPSDE,
    VESDE
)
from src.models.blocks import (
    RefineNet,
    WideResnet
)
from src.sampling import sampling_fn

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
        noise_removal,
        **kwargs
):
    sampler = sampling_fn(
        predictor_name=predictor_name,
        corrector_name=corrector_name,
        sde=sde,
        shape=shape,
        eps=sampling_eps,
        snr=snr,
        n_steps=n_steps,
        noise_removal=noise_removal
    )
    return sampler


def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
      Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
      Returns:
        A score function.
      """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t):
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn