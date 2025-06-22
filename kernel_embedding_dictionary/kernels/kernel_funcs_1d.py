# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import factorial

from kernel_embedding_dictionary.utils import scaled_diff


def expquad_kernel_func_1d(x1: float, x2: float, ell: float) -> float:
    diff = 0.5 * scaled_diff(x1, x2, ell, np.sqrt(2))
    kernel_value = np.exp(-(diff**2))
    return kernel_value


def matern_kernel_func_1d(x1: float, x2: float, ell: float, nu: float) -> float:
    n = int(nu)
    ks = np.arange(n + 1)
    coefs = factorial(n) / factorial(2 * n) * np.power(2, ks) * factorial(n + ks) / factorial(ks) / factorial(n - ks)
    abs_diff = np.sqrt(2 * nu) * abs(scaled_diff(x1, x2, ell, 1))
    abs_diff_powers = np.power(abs_diff, n - ks)
    kernel_value = np.exp(-abs_diff) * np.sum(coefs * abs_diff_powers)
    return kernel_value


def matern12_kernel_func_1d(x1: float, x2: float, ell: float, nu: float) -> float:
    diff = scaled_diff(x1, x2, ell, 1)
    kernel_value = np.exp(-abs(diff))
    return kernel_value


def matern32_kernel_func_1d(x1: float, x2: float, ell: float, nu: float) -> float:
    abs_diff = np.sqrt(3) * abs(scaled_diff(x1, x2, ell, 1))
    kernel_value = (1 + abs_diff) * np.exp(-abs_diff)
    return kernel_value


def matern52_kernel_func_1d(x1: float, x2: float, ell: float, nu: float) -> float:
    abs_diff = np.sqrt(5) * abs(scaled_diff(x1, x2, ell, 1))
    kernel_value = (1 + abs_diff + abs_diff**2 / 3) * np.exp(-abs_diff)
    return kernel_value


def matern72_kernel_func_1d(x1: float, x2: float, ell: float, nu: float) -> float:
    abs_diff = np.sqrt(7) * abs(scaled_diff(x1, x2, ell, 1))
    kernel_value = (1 + abs_diff + 2 * abs_diff**2 / 5 + abs_diff**3 / 15) * np.exp(-abs_diff)
    return kernel_value
