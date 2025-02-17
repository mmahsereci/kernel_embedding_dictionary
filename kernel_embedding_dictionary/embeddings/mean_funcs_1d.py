# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf

from kernel_embedding_dictionary.utils import scaled_diff


def expquad_lebesgue_mean_func_1d(x: np.ndarray, ell: float, lb: float, ub: float, density: float) -> np.ndarray:
    print(ell, lb, ub, density)
    erf_lo = erf(scaled_diff(lb, x, ell))
    erf_up = erf(scaled_diff(ub, x, ell))
    kernel_mean = np.sqrt(np.pi / 2.0) * ell * (erf_up - erf_lo)
    return density * kernel_mean.reshape(-1)
