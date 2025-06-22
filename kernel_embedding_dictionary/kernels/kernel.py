# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc
from typing import List, Union

import numpy as np


class UnivariateKernel(abc.ABC):

    @property
    @abc.abstractmethod
    # TODO: turn into absract property
    def param_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def _evaluate_pair(self, x1: float, x2: float) -> float:
        """use pass in case _evaluate is overridden"""
        pass

    def _evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """override for efficiency. x1 and x2 have shape (n1, ) and (n2, )."""
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._evaluate_pair(x1[i], x2[j])
        return K

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """x1 and x2 have shape (n1, ) and (n2, )."""
        if (len(x1.shape)) != 1 or (len(x2.shape) != 1):
            raise ValueError(f"x1 ({x1.shape}) and x2 ({x2.shape}) must be one-dimensional arrays.")

        return self._evaluate(x1, x2)


class ProductKernel(abc.ABC):
    def __init__(self, name: str, kernel_list: List[UnivariateKernel]):
        self.name = name
        self._kernels = kernel_list

    @property
    def ndim(self) -> int:
        return len(self._kernels)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def get_param_dict_from_dim(self, dim: int) -> dict:
        return self._kernels[dim].param_dict

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if (len(x1.shape) != 2) or (len(x2.shape) != 2):
            raise ValueError(f"x1 or x2 have wrong shape.")

        n1, d1 = x1.shape
        n2, d2 = x2.shape

        if d1 != d2:
            raise ValueError(f"x1 ({d1}) and x2 ({d2}) must have matching dimensionality.")

        if d1 != self.ndim:
            raise ValueError(f"x1 and x2 have wrong dimensionality ({d1}).")

        K = np.ones([n1, n2])
        for dim, k in enumerate(self._kernels):
            K *= k.evaluate(x1[:, dim], x2[:, dim])
        return K
