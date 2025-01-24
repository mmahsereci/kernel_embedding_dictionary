# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc
from typing import List

import numpy as np


class UnivariateKernel(abc.ABC):
    @abc.abstractmethod
    def get_param_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def _evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if (len(x1.shape)) != 1 or (len(x2.shape) != 1):
            raise ValueError(f"x1 ({x1.shape}) and x2 ({x2.shape}) must be one-dimensional arrays.")

        return self._evaluate(x1, x2)


class ProductKernel(abc.ABC):
    def __init__(self, name: str, kernel_list: List[UnivariateKernel, ...]):
        self.name = name
        self._kernels = kernel_list

    @property
    def ndim(self):
        return len(self._kernels)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def get_param_dict_from_dim(self, dim: int) -> dict:
        return self._kernels[dim].get_param_dict()

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1, d1 = x1.shape
        n2, d2 = x2.shape

        if d1 != d2:
            raise ValueError(f"x1 ({d1}) and x2 ({d2}) must have matching dimensionality.")

        K = np.ones([n1, n2])
        for k in self._kernels:
            K *= k.evaluate(x1, x2)
        return K
