# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Callable

import numpy as np

from ..kernels import ProductKernel
from ..measures import ProductMeasure
from .mean_funcs_1d import (
    expquad_gaussian_mean_func_1d,
    expquad_lebesgue_mean_func_1d,
    matern12_gaussian_mean_func_1d,
    matern12_lebesgue_mean_func_1d,
    matern32_gaussian_mean_func_1d,
    matern32_lebesgue_mean_func_1d,
    matern52_lebesgue_mean_func_1d,
    matern72_lebesgue_mean_func_1d,
    wendland0_gaussian_mean_func_1d,
    wendland0_lebesgue_mean_func_1d,
    wendland2_gaussian_mean_func_1d,
)


class KernelEmbedding:
    def __init__(self, kernel: ProductKernel, measure: ProductMeasure):
        if kernel.ndim != measure.ndim:
            raise ValueError(f"kernel ({kernel.ndim}) and measure ({measure.ndim}) need to have same dimensionality.")

        self.ndim = kernel.ndim
        self._kernel = kernel
        self._measure = measure

        # kernel and measure must be set first
        self._mean_func_1d = self._get_1d_funcs()

    def __str__(self) -> str:
        return f"Kernel embedding for\n\n{self._kernel.__str__()}\n\nand\n\n{self._measure.__str__()}"

    def __repr__(self) -> str:
        return f"Kernel embedding for {self._kernel.__repr__()} and {self._measure.__repr__()}."

    def mean(self, x: np.ndarray) -> np.ndarray:

        e_msg = f"x has wrong shape {x.shape}. Perhaps the dimensionality does not match the kernel embedding."
        if len(x.shape) != 2:
            raise ValueError(e_msg)
        if self.ndim != x.shape[1]:
            raise ValueError(e_msg)

        kernel_mean = np.ones(x.shape[0])
        for dim in range(x.shape[1]):
            params_dim = {**self._kernel.get_param_dict_from_dim(dim), **self._measure.get_param_dict_from_dim(dim)}
            # TODO: remove after debugging
            print(params_dim)
            kernel_mean *= self._mean_func_1d(x[:, dim], **params_dim)
        return kernel_mean

    def _get_1d_funcs(self) -> Callable:

        mean_func_1d_dict = {
            "expquad-lebesgue": expquad_lebesgue_mean_func_1d,
            "expquad-gaussian": expquad_gaussian_mean_func_1d,
            "matern12-lebesgue": matern12_lebesgue_mean_func_1d,
            "matern12-gaussian": matern12_gaussian_mean_func_1d,
            "matern32-lebesgue": matern32_lebesgue_mean_func_1d,
            "matern32-gaussian": matern32_gaussian_mean_func_1d,
            "matern52-lebesgue": matern52_lebesgue_mean_func_1d,
            "matern72-lebesgue": matern72_lebesgue_mean_func_1d,
            "wendland0-lebesgue": wendland0_lebesgue_mean_func_1d,
            "wendland0-gaussian": wendland0_gaussian_mean_func_1d,
            "wendland2-gaussian": wendland2_gaussian_mean_func_1d,
        }

        mean_func_1d = mean_func_1d_dict.get(self._kernel.name + "-" + self._measure.name, None)

        if not mean_func_1d:
            raise ValueError(f"kernel embedding unknown.")

        return mean_func_1d
