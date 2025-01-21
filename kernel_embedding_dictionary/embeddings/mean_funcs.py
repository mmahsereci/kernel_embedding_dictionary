# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from scipy.special import erf

from ..kernels import ExpQuadKernel
from ..measures import LebesgueMeasure
from .utils import scaled_vector_diff


def expquad_lebesgue_mean(x: np.ndarray, k: ExpQuadKernel, m: LebesgueMeasure) -> np.ndarray:
    ell = k.ell
    lb = m.lb[None, :]
    ub = m.ub[None, :]
    erf_lo = erf(scaled_vector_diff(lb, x, ell))
    erf_up = erf(scaled_vector_diff(ub, x, ell))
    kernel_mean = (np.sqrt(np.pi / 2.0) * k.ell * (erf_up - erf_lo)).prod(axis=1)
    return m.density * kernel_mean.reshape(-1)
