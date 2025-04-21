# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.special import factorial

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

def matern_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    n = int(nu)
    cs = np.zeros(n + 1)
    for m in range(n + 1):
        iss = np.arange(n - m + 1)
        cs[m] = np.sum(np.power(2, n - iss) * factorial(n + iss) / factorial(iss)) / factorial(m)
    alpha = ell / np.sqrt(2 * nu)
    ms = np.arange(n + 1)
    x = x.reshape(len(x.reshape(-1)), 1)
    x_lb = (x - lb) / alpha
    Q_lb = np.exp(-x_lb.reshape(-1)) * np.sum(cs * np.power(x_lb, ms), 1)
    x_ub = (ub - x) / alpha
    Q_ub = np.exp(-x_ub.reshape(-1)) * np.sum(cs * np.power(x_ub, ms), 1)
    kernel_mean = alpha * factorial(n) / factorial(2*n) * (2 * cs[0] - Q_lb - Q_ub)
    return density * kernel_mean

def matern12_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    exp_lb_x = np.exp(scaled_diff(lb, x, ell, 1))
    exp_x_ub = np.exp(scaled_diff(x, ub, ell, 1))
    kernel_mean = ell * (2.0 - exp_lb_x - exp_x_ub)
    return density * kernel_mean.reshape(-1)

def matern12_gaussian_mean_func_1d(x: np.ndarray, ell: float, nu: float, mean: float, variance: float) -> np.ndarray:

    arg_var = scaled_diff(x, mean, np.sqrt(variance), 1)
    arg_ell = scaled_diff(x, mean, ell, 1)

    cdf_term_1 = norm.cdf(-arg_var - np.sqrt(variance) / ell)
    cdf_term_2 = norm.cdf(arg_var - np.sqrt(variance) / ell)

    exp_term_1 = np.exp(variance / (2 * ell**2) + arg_ell)
    exp_term_2 = np.exp(variance / (2 * ell**2) - arg_ell)
    return exp_term_1 * cdf_term_1 + exp_term_2 * cdf_term_2


def matern32_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    diff_x_ub = np.sqrt(3) * scaled_diff(x, ub, ell, 1)
    diff_lb_x = np.sqrt(3) * scaled_diff(lb, x, ell, 1)
    exp_term_1 = np.exp(diff_x_ub) * (ub + 2.0 * ell / np.sqrt(3) - x)
    exp_term_2 = np.exp(diff_lb_x) * (x + 2.0 * ell / np.sqrt(3) - lb)
    kernel_mean = 4.0 * ell / np.sqrt(3) - exp_term_1 - exp_term_2
    return density * kernel_mean.reshape(-1)


def matern32_gaussian_mean_func_1d(x: np.ndarray, ell: float, nu: float, mean: float, variance: float) -> np.ndarray:

    mu_1 = mean - np.sqrt(3) * variance / ell
    mu_2 = mean + np.sqrt(3) * variance / ell

    def d_arg(x1: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return scaled_diff(x1, mu, ell, 1 / np.sqrt(3))

    exp_term_1 = np.exp(3 * variance / (2 * ell**2) + d_arg(x, mean))
    exp_term_2 = np.exp(3 * variance / (2 * ell**2) - d_arg(x, mean))

    cdf_term_1 = norm.cdf(scaled_diff(mu_1, x, np.sqrt(variance), 1)) * (1 - d_arg(x, mu_1))
    cdf_term_2 = norm.cdf(scaled_diff(x, mu_2, np.sqrt(variance), 1)) * (1 + d_arg(x, mu_2))

    norm_term_1 = norm.pdf(x, mu_1, np.sqrt(variance)) * np.sqrt(3) * variance / ell
    norm_term_2 = norm.pdf(x, mu_2, np.sqrt(variance)) * np.sqrt(3) * variance / ell

    kernel_mean = exp_term_1 * (cdf_term_1 + norm_term_1) + exp_term_2 * (cdf_term_2 + norm_term_2)
    return kernel_mean.reshape(-1)


def matern52_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    diff_x_ub = np.sqrt(5) * scaled_diff(x, ub, ell, 1)
    diff_lb_x = np.sqrt(5) * scaled_diff(lb, x, ell, 1)

    def exp_term(diff: np.ndarray) -> np.ndarray:
        return np.exp(diff) * (8.0 - 5.0 * diff + diff**2)

    prefactor = ell / (3 * np.sqrt(5))
    kernel_mean = prefactor * (16.0 - exp_term(diff_x_ub) - exp_term(diff_lb_x))
    return density * kernel_mean.reshape(-1)


def matern72_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, nu: float, lb: float, ub: float, density: float
) -> np.ndarray:
    diff_x_ub = np.sqrt(7) * scaled_diff(x, ub, ell, 1)
    diff_lb_x = np.sqrt(7) * scaled_diff(lb, x, ell, 1)

    def exp_term(diff: np.ndarray) -> np.ndarray:
        return np.exp(diff) * (48.0 - 33.0 * diff + 9.0 * diff**2 - diff**3)

    prefactor = ell / (15 * np.sqrt(7))
    kernel_mean = prefactor * (96.0 - exp_term(diff_x_ub) - exp_term(diff_lb_x))
    return density * kernel_mean.reshape(-1)
