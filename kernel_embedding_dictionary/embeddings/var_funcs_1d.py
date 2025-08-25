# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf


def expquad_lebesgue_var_func_1d(ell: float, lb: float, ub: float, density: float) -> np.ndarray:
    """Compute the expected mean function for the exponential quadratic kernel with respect to the Lebesgue measure in 1D.
    
    Args:
        ell: The length scale parameter.
        lb: The lower bound of the integration interval.
        ub: The upper bound of the integration interval.
        density: The density of the Lebesgue measure.

    Returns:
        The expected mean function value.
    """
    exp_term = ell * np.sqrt(2/np.pi) * (np.exp(-0.5 * ((ub - lb) / ell) ** 2) - 1)
    erf_term = (ub - lb) * (erf((ub - lb) / (ell * np.sqrt(2))))
    return np.sqrt(2 * np.pi) * ell * density**2 * (exp_term + erf_term)