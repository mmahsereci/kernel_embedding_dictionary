# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Callable

import numpy as np

from ..kernels import Kernel
from ..measures import Measure


class KernelEmbedding:
    def __init__(self, kernel: Kernel, measure: Measure, mean_func: Callable):
        if kernel.ndim != measure.ndim:
            raise ValueError(f"kernel ({kernel.ndim}) and measure ({measure.ndim}) need to have same dimensionality.")

        self._kernel = kernel
        self._measure = measure
        self._mean_func = mean_func

        self.ndim = kernel.ndim

    def mean(self, x: np.ndarray) -> np.ndarray:

        if (self.ndim != x.shape[1]) or (len(x.shape) != 2):
            raise ValueError(
                f"x has wrong shape {x.shape}. Perhaps the dimensionality does not match the kernel embedding."
            )

        return self._mean_func(x, self._kernel, self._measure)

    def __str__(self) -> str:

        return f"Kernel embedding for\n\n{self._kernel.__str__()}\n\nand\n\n{self._measure.__str__()}"

    def __repr__(self) -> str:
        return f"Kernel embedding for {self._kernel.__repr__()} and {self._measure.__repr__()}."
