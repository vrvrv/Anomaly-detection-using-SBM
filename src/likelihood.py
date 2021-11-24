import torch
import numpy as np
import src.models.utils as mutils
from scipy import integrate


def get_div_fn(fn):
    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(
        sde, hutchinson_type='Rademacher', rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5
):
    def drift_fn(model, x, t):
        score_fn = mutils.get_score_fn(sde, model, train=False)
        rsde = sde.reverse(score_fn, probability_flow=True)

        reverse_drift, reverse_diffusion = rsde.sde(x, t)

        return reverse_drift

    def div_fn(model, x, t, noise):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn = drift_fn(model, x, t)
            fn_eps = torch.sum(fn * noise)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return (grad_fn_eps * eps).sum(dim=tuple(range(1, len(x.shape))))

    def likelihood_fn(model, data: np.array):
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = torch.from_numpy(x[:-shape[0]].reshape(shape)).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = drift_fn(model, sample, vec_t).detach().cpu().numpy().reshape((-1, ))
                logp_grad =div_fn(model, sample, vec_t, epsilon).detach().cpu().numpy().reshape((-1, ))
                return np.concatenate([drift, logp_grad], axis=0)

            data_flatten = data.detach().cpu().numpy().reshape((-1, ))

            init = np.concatenate([data_flatten, np.zeros((shape[0], ))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)

            zp = solution.y[:, -1]

            z = torch.from_numpy(
                zp[:-shape[0]].reshape(shape)
            ).to(data.device).type(torch.float32)

            delta_logp = torch.from_numpy(
                zp[-shape[0]:].reshape(shape[0], )
            ).to(data.device).type(torch.float32)
            prior_logp = sde.prior_logp(z)

            bpd = (prior_logp + delta_logp) / (np.log(2) * np.prod(shape[1:]))

        return bpd

    return likelihood_fn