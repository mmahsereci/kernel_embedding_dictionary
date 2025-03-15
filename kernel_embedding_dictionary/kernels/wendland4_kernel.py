# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import List, Optional

import numpy as np

from ..utils import scaled_diff
from .kernel import ProductKernel, UnivariateKernel


class Wendland4KernelUni(UnivariateKernel):
    def __init__(self, ell: float):

        if ell <= 0:
            raise ValueError(f"ell ({ell}) must be positive")

        self.ell = ell
        self.order = 4

    def get_param_dict(self) -> dict:
        return {"ell": self.ell, "order": self.order}

    def _evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                abs_diff = abs(scaled_diff(x1[i], x2[j], self.ell, 1))
                K[i, j] = max(0, (1 - abs_diff))**5 * (1 + 5 * abs_diff + 8 * abs_diff**2)
        return K


class Wendland4Kernel(ProductKernel):
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

        kernels = [Wendland4KernelUni(ell=elli) for elli in ell]
        super().__init__(name="Wendland4", kernel_list=kernels)  # sets name, ndim and kernel list

    @property
    def ell(self) -> List[float]:
        return [k.ell for k in self._kernels]

    @property
    def order(self) -> float:
        return 4

    def __str__(self) -> str:
        return f"Wendland4 kernel \n" f"dimensionality: {self.ndim} \n" f"lengthscales: {list(self.ell)}"

    def __repr__(self) -> str:
        return "Wendland4 kernel"
