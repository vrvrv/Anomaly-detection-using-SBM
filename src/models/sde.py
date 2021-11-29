import abc
import torch
import numpy as np


class SDE(abc.ABC):
    def __init__(self, N):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super(SDE, self).__init__()
        self.N = N

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x_t|x_0)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z, mask=None):
        """
        Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.
        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.
        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """
        Create the reverse-time SDE/ODE

        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class ReverseSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return ReverseSDE()


class VPSDE(SDE):
    def __init__(
            self,
            beta_0: float = 0.1,
            beta_1: float = 20,
            N: int = 1000,
            **kwargs
    ):
        super().__init__(N)

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.N = N
        self.discrete_betas = torch.linspace(beta_0 / N, beta_1 / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = - .5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = - .25 * t ** 2 * (self.beta_1 - self.beta_0) - .5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z, mask=None):
        """
        z : batch_size x d1 x ... x dM
        N = d1 * d2 * ... * dM
        """
        if mask is None:
            mask = torch.ones_like(z)

        N = 3 * mask.sum(tuple(range(1, len(mask.shape))))
        logps = - N / 2. * np.log(2 * np.pi) - torch.sum((z * mask) ** 2, dim=(1, 2, 3)) / 2.

        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

class subVPSDE(SDE):
    def __init__(
            self,
            beta_0: float = 0.1,
            beta_1: float = 20,
            N: int = 1000,
            **kwargs
    ):
        super(subVPSDE, self).__init__(N)

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = - .5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z, mask=None):
        if mask is None:
            mask = torch.ones_like(z)

        N = 3 * mask.sum(tuple(range(1, len(mask.shape))))

        return - N / 2. * np.log(2 * np.pi) - torch.sum((z * mask) ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
    def __init__(
            self,
            sigma_0: float = 0.01,
            sigma_1: float = 50,
            N: int = 1000,
            **kwargs
    ):
        super(VESDE, self).__init__(N)
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_0), np.log(self.sigma_1), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_0 * (self.sigma_1 / self.sigma_0) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_1) - np.log(self.sigma_0)),
                device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_0 * (self.sigma_1 / self.sigma_0) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_1

    def prior_logp(self, z, mask=None):
        if mask is None:
            mask = torch.ones_like(z)

        N = 3 * mask.sum(tuple(range(1, len(mask.shape))))

        return - N / 2. * np.log(2 * np.pi * self.sigma_1 ** 2) \
               - torch.sum((z * mask) ** 2, dim=(1, 2, 3)) / (2 * self.sigma_1 ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G