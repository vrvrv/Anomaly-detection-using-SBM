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
    def prior_logp(self, z):
        """
        Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.
        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass


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

    def prior_logp(self, z):
        """
        z : batch_size x d1 x ... x dM
        N = d1 * d2 * ... * dM
        """
        N = np.prod(z.shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps


class subVPSDE(SDE):
    def __init__(
            self,
            beta_0: float = 0.1,
            beta_1: float = 20,
            N: int = 2000,
            **kwargs
    ):
        super(subVPSDE, self).__init__(N)

        self.beta_0 = beta_0
        self.beta_1 = beta_1

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

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


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

    def prior_logp(self, z):
        N = np.prod(z.shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_1 ** 2) \
               - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_1 ** 2)
