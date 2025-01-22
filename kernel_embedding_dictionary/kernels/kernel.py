# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc

import numpy as np


class Kernel(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1, d1 = x1.shape
        n2, d2 = x2.shape

        if d1 != d2:
            raise ValueError(f"x1 ({d1}) and x2 ({d2}) must have matching dimensionality.")


