# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import List, Optional

import numpy as np

from ..utils import scaled_diff
from .kernel import ProductKernel, UnivariateKernel


class Matern52KernelUni(UnivariateKernel):
    def __init__(self, ell: float):

        if ell <= 0:
            raise ValueError(f"ell ({ell}) must be positive")

        self.ell = ell
        self.nu = 1.5

    def get_param_dict(self) -> dict:
        return {"ell": self.ell, "nu": self.nu}

    def _evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                abs_diff = np.sqrt(5) * abs(scaled_diff(x1[i], x2[j], self.ell, 1))
                K[i, j] = (1 + abs_diff + abs_diff**2/3) * np.exp(-abs_diff)
        return K


class Matern52Kernel(ProductKernel):
    def __init__(self, config: Optional[dict] = None):
        """

        :param config: needs to contain ells and/or ndim
        """

        if config is None:
            config = {}

        # dimensionality and bounds
        ell = config.get("lengthscales", [1.0])
        ndim = config.get("ndim", None)

        if ndim is not None:
            if (ndim > 1) and (len(ell) == 1):
                ell = ndim * [ell[0]]
            else:
                if ndim != len(ell):
                    raise ValueError(f"ndim ({ndim}) and dimensionality of lengthscales ({len(ell)}) does not match.")

        kernels = [Matern52KernelUni(ell=elli) for elli in ell]
        super().__init__(name="matern32", kernel_list=kernels)  # sets name, ndim and kernel list

    @property
    def ell(self) -> List[float]:
        return [k.ell for k in self._kernels]

    @property
    def nu(self) -> float:
        return 1.5

    def __str__(self) -> str:
        return f"Matern5/2 kernel \n" f"dimensionality: {self.ndim} \n" f"lengthscales: {list(self.ell)}"

    def __repr__(self) -> str:
        return "Matern5/2 kernel"
