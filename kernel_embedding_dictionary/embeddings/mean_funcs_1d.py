# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf, factorial
from scipy.stats import norm

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
    kernel_mean = alpha * factorial(n) / factorial(2 * n) * (2 * cs[0] - Q_lb - Q_ub)
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


def wendland0_lebesgue_mean_func_1d(
    x: np.ndarray, ell: float, order: int, lb: float, ub: float, density: float
) -> np.ndarray:
    kernel_mean = np.zeros_like(x)

    mask1 = (ub >= (x + ell)) & ((lb + ell) < x)
    mask2 = (ub >= (x + ell)) & ((lb + ell) >= x)
    mask3 = (ub < (x + ell)) & ((lb + ell) < x)
    mask4 = ~(mask1 | mask2 | mask3)

    kernel_mean[mask1] = ell
    kernel_mean[mask2] = (2 * x[mask2] * (lb + ell) + ell**2 - lb**2 - 2 * lb * ell - x[mask2] ** 2) / (2 * ell)
    kernel_mean[mask3] = (2 * ub * (ell + x[mask3]) + ell**2 - ub**2 - 2 * x[mask3] * ell - x[mask3] ** 2) / (2 * ell)
    kernel_mean[mask4] = (
        2 * (ub * ell + ub * x[mask4] + lb * x[mask4]) - lb**2 - ub**2 - 2 * (lb * ell + x[mask4] ** 2)
    ) / (2 * ell)

    return density * kernel_mean.reshape(-1)


def wendland0_gaussian_mean_func_1d(x: np.ndarray, ell: float, order: int, mean: float, variance: float) -> np.ndarray:

    if mean != 0.0:
        raise ValueError("Only mean=0 is supported.")

    s = np.sqrt(2 * variance)

    def phi(x: np.ndarray) -> np.ndarray:
        """Unnormalized Gaussian."""
        return np.exp(-(x**2) / s**2)

    def Phi(x: np.ndarray) -> np.ndarray:
        """Scaled error function."""
        return erf(x / s)

    erf_terms = (ell - x) * Phi(ell - x) + (ell + x) * Phi(ell + x) - 2 * x * Phi(x)
    gauss_terms = (phi(ell - x) + phi(ell + x) - 2 * phi(x)) * s / np.sqrt(np.pi)
    kernel_mean = (erf_terms + gauss_terms) / (2 * ell)

    return kernel_mean.reshape(-1)


def wendland2_gaussian_mean_func_1d(x: np.ndarray, ell: float, order: int, mean: float, variance: float) -> np.ndarray:

    if mean != 0.0:
        raise ValueError("Only mean=0 is supported.")

    s = np.sqrt(2 * variance)

    def phi(x: np.ndarray) -> np.ndarray:
        """Unnormalized Gaussian."""
        return np.exp(-(x**2) / s**2)

    def Phi(x: np.ndarray) -> np.ndarray:
        """Scaled error function."""
        return erf(x / s)

    def dot_product(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
        """Dot product of two lists."""
        return sum([a_i * b_i for a_i, b_i in zip(a, b)])

    # Exponential function terms
    term_1 = (phi(x + ell) + phi(x - ell)) * (ell**3 - ell * (7 * variance + 5 * x**2))
    term_2 = (phi(x + ell) - phi(x - ell)) * (ell**2 * x + 3 * x * (5 * variance + x**2))
    term_3 = phi(x) * 16 * ell * (2 * variance + x**2)

    exp_term = (term_1 - term_2 + term_3) * s / np.sqrt(np.pi)

    # Coefficients and terms for the error function
    erf_prefac_terms = [
        ell**4,
        6 * ell**2 * (x**2 + variance),
        8 * ell * (3 * variance * x + x**3),
        3 * (3 * variance**2 + 6 * variance * x**2 + x**4),
    ]
    signs_p = [1, -1, -1, -1]
    signs_m = [1, -1, 1, -1]

    erf_term = (
        dot_product(signs_p, erf_prefac_terms) * Phi(ell + x)
        + dot_product(signs_m, erf_prefac_terms) * Phi(ell - x)
        + 16 * ell * x * (3 * variance + x**2) * Phi(x)
    )

    kernel_mean = (exp_term + erf_term) / (2 * ell**4)
    return kernel_mean.reshape(-1)
