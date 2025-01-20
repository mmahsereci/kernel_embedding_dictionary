# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Callable

import numpy as np

from ..kernels import Kernel
from ..measures import Measure


class KernelEmbedding:
    def __init__(self, kernel: Kernel, measure: Measure, eval_func: Callable):
        self._kernel = kernel
        self._measure = measure
        self._evaluate = eval_func

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x, self._kernel, self._measure)

    def __str__(self) -> str:

        return f"Kernel embedding for\n\n{self._kernel.__str__()}\n\nand\n\n{self._measure.__str__()}"

    def __repr__(self) -> str:
        return f"Kernel embedding for {self._kernel.__repr__()} and {self._measure.__repr__()}."
