# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Callable

import numpy as np

from ..kernels import ProductKernel, ExpQuadKernel
from ..measures import ProductMeasure, LebesgueMeasure

from .mean_funcs import expquad_lebesgue_mean_func_1d


class KernelEmbedding:
    def __init__(self, kernel: ProductKernel, measure: ProductMeasure):
        if kernel.ndim != measure.ndim:
            raise ValueError(f"kernel ({kernel.ndim}) and measure ({measure.ndim}) need to have same dimensionality.")

        self.ndim = kernel.ndim
        self._kernel = kernel
        self._measure = measure

        # kernel and measure must be set first
        self._set_embedding_specific_1d_funcs()

    def __str__(self) -> str:
        return f"Kernel embedding for\n\n{self._kernel.__str__()}\n\nand\n\n{self._measure.__str__()}"

    def __repr__(self) -> str:
        return f"Kernel embedding for {self._kernel.__repr__()} and {self._measure.__repr__()}."

    def mean(self, x: np.ndarray) -> np.ndarray:

        if (self.ndim != x.shape[1]) or (len(x.shape) != 2):
            raise ValueError(
                f"x has wrong shape {x.shape}. Perhaps the dimensionality does not match the kernel embedding."
            )

        kernel_mean = np.ones(x.shape[0])
        for dim in range(x.shape[1]):
            params_dim = {**self._kernel.get_param_dict_from_dim(dim), **self._measure.get_param_dict_from_dim(dim)}
            print(params_dim)  # TODO: remove after debugging
            kernel_mean *= self._mean_func_1d(x[:, dim], **params_dim)
        return kernel_mean

    def _set_embedding_specific_1d_funcs(self):

        if self._kernel.name == "expquad":
            if self._measure.name == "lebesgue":
                mean_func_1d = expquad_lebesgue_mean_func_1d

        self._mean_func_1d = mean_func_1d
