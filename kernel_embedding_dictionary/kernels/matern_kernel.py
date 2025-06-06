# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import List, Optional

import numpy as np
from scipy.special import factorial

from ..utils import scaled_diff
from .kernel import ProductKernel, UnivariateKernel


class MaternNu2KernelUni(UnivariateKernel):
    def __init__(self, nu: float, ell: float):

        if ell <= 0:
            raise ValueError(f"ell ({ell}) must be positive")

        if not ( int(nu + 0.5) == nu + 0.5 and nu > 0 ):
            raise ValueError(f"only kernels and embeddings for positive half-integer nu ({nu}) are implemented")

        self.ell = ell
        self.nu = nu

    def get_param_dict(self) -> dict:
        return {"ell": self.ell, "nu": self.nu}

    def _evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros([n1, n2])
        n = int(self.nu)
        ks = np.arange(n+1)
        coefs =  factorial(n) / factorial(2 * n) * np.power(2, ks) * factorial(n + ks) / factorial(ks) / factorial(n - ks)
        for i in range(n1):
            for j in range(n2):
                abs_diff = np.sqrt( 2 *self.nu) * abs(scaled_diff(x1[i], x2[j], self.ell, 1))
                abs_diff_powers = np.power(abs_diff, n - ks)
                K[i, j] = np.exp(-abs_diff) * np.sum(coefs * abs_diff_powers)
        return K


class MaternNu2Kernel(ProductKernel):
    def __init__(self, config: Optional[dict] = None):
        """

        :param config: needs to contain nu, ells and/or ndim
        """

        if config is None:
            config = {}

        # dimensionality and bounds
        nu = config.get("nu", 2.5)
        ell = config.get("lengthscales", [1.0])
        ndim = config.get("ndim", None)

        if ndim is not None:
            if (ndim > 1) and (len(ell) == 1):
                ell = ndim * [ell[0]]
            else:
                if ndim != len(ell):
                    raise ValueError(f"ndim ({ndim}) and dimensionality of lengthscales ({len(ell)}) does not match.")

        kernels = [MaternNu2KernelUni(nu=nu, ell=elli) for elli in ell]
        super().__init__(name="matern", kernel_list=kernels)  # sets name, ndim and kernel list

        self._nu = nu

    @property
    def ell(self) -> List[float]:
        return [k.ell for k in self._kernels]

    @property
    def nu(self) -> float:
        return self._nu 

    def __str__(self) -> str:
        return f"Matern kernel \n" f"order: {self.nu} \n" f"dimensionality: {self.ndim} \n" f"lengthscales: {list(self.ell)}"

    def __repr__(self) -> str:
        return "Matern kernel"