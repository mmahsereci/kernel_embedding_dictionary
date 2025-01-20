# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Optional

import numpy as np

from .kernel import Kernel


class ExpQuadKernel(Kernel):
    def __init__(self, config: Optional[dict] = None):
        """

        :param config: needs to contain bounds, (optionalL: ndim), normalized
        """
        # this will yield the standard ExpQuad kernel in 1D
        if config is None:
            config = {}

        # dimensionality and bounds
        ell = config.get("lengthscales", [1.0])
        ndim = config.get("ndim", None)

        if ndim is None:
            ndim = len(ell)
        else:
            if (ndim > 1) and (len(ell) == 1):
                ell = ndim * [ell[0]]
            else:
                if ndim != len(ell):
                    raise ValueError(f"ndim ({ndim}) and dimensionality of lengthscales ({len(ell)}) does not match.")

        self.ndim = ndim
        self.ell = np.array(ell)

    def k(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # TODO: implement this efficiently
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                diff = (x1[i, :] - x2[j, :]) / self.ell
                K[i, j] = np.exp(np.sum(-0.5 * diff**2))
        return K

    def __str__(self) -> str:
        return f"Exponentiated quadratic kernel \n" \
               f"dimensionality: {self.ndim} \n" \
               f"lengthscales: {list(self.ell)}"

    def __repr__(self) -> str:
        return "exponentiated quadratic kernel"
