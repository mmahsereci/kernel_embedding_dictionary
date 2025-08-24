# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import List, Optional

from .kernel import ProductKernel, UnivariateKernel
from .kernel_funcs_1d import matern72_kernel_func_1d


class Matern72KernelUni(UnivariateKernel):
    def __init__(self, ell: float):

        if ell <= 0:
            raise ValueError(f"ell ({ell}) must be positive")

        self.ell = ell
        self.nu = 3.5

        self._kernel_func = matern72_kernel_func_1d

    @property
    def param_dict(self) -> dict:
        return {"ell": self.ell, "nu": self.nu}

    def _evaluate_pair(self, x1: float, x2: float) -> float:
        return self._kernel_func(x1, x2, **self.param_dict)


class Matern72Kernel(ProductKernel):
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

        kernels = [Matern72KernelUni(ell=elli) for elli in ell]
        super().__init__(name="matern72", kernel_list=kernels)  # sets name, ndim and kernel list

    @property
    def ell(self) -> List[float]:
        return [k.ell for k in self._kernels]

    @property
    def nu(self) -> float:
        return 3.5

    def __str__(self) -> str:
        return f"Matern7/2 kernel \n" f"dimensionality: {self.ndim} \n" f"lengthscales: {list(self.ell)}"

    def __repr__(self) -> str:
        return "Matern7/2 kernel"
