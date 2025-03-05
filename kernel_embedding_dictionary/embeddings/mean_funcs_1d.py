# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf

from kernel_embedding_dictionary.utils import scaled_diff


def expquad_lebesgue_mean_func_1d(x: np.ndarray, ell: float, lb: float, ub: float, density: float) -> np.ndarray:
    erf_lo = erf(scaled_diff(lb, x, ell, np.sqrt(2)))
    erf_up = erf(scaled_diff(ub, x, ell, np.sqrt(2)))
    kernel_mean = np.sqrt(np.pi / 2.0) * ell * (erf_up - erf_lo)
    return density * kernel_mean.reshape(-1)


def expquad_gaussian_mean_func_1d(x: np.ndarray, ell: float, mean: float, variance: float) -> np.ndarray:
    factor = np.sqrt(ell**2 / (ell**2 + variance))
    scaled_norm_sq = np.power(scaled_diff(x, mean, np.sqrt(ell**2 + variance), np.sqrt(2)), 2)
    return factor * np.exp(-scaled_norm_sq).reshape(-1)


def matern12_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    print(ell, lb, ub, density)
    exp_lb_x = np.exp(scaled_diff(lb, x, ell, 1))
    exp_x_ub = np.exp(scaled_diff(x, ub, ell, 1))
    kernel_mean = ell * (2.0 - exp_lb_x - exp_x_ub)
    return density * kernel_mean.reshape(-1)
