import numpy as np


def get_sigmas(sigma_0: float, sigma_1:float, num_scales: int):
    """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a numpy arrary of noise levels
  """
    sigmas = np.exp(
        np.linspace(np.log(sigma_1), np.log(sigma_0), num_scales)
    )

    return sigmas
