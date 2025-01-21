# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Optional

import numpy as np

from .measure import Measure


class LebesgueMeasure(Measure):
    def __init__(self, config: Optional[dict] = None):
        """

        :param config: needs to contain bounds, (optionalL: ndim), normalized
        """
        # this will yield the standard Lebesgue measure in 1D
        if config is None:
            config = {}

        # dimensionality and bounds
        bounds = config.get("bounds", [(0, 1)])
        ndim = config.get("ndim", None)

        if ndim is None:
            ndim = len(bounds)
        else:
            if (ndim > 1) and (len(bounds) == 1):
                bounds = ndim * [bounds[0]]
            else:
                if ndim != len(bounds):
                    raise ValueError(f"ndim ({ndim}) and dimensionality fo bounds ({len(bounds)}) does not match.")

        self.ndim = ndim
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])

        # normalization and density
        normalize = config.get("normalize", False)
        density = 1.0
        if normalize:
            density = 1.0 / np.prod(self.ub - self.lb)
        self.normalize = normalize
        self.density = density

    @property
    def bounds(self):
        return [(lb, ub) for lb, ub in zip(self.lb, self.ub)]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        # TODO: does not account for x that may lie outside of the bounds (in which case the pdf should be zero)
        return np.full(x.shape[0], self.density)

    def __str__(self) -> str:
        return (
            f"Lebesgue measure \n"
            f"dimensionality: {self.ndim} \n"
            f"normalized: {self.normalize}\n"
            f"density: {self.density}\n"
            f"bounds: {self.bounds}"
        )

    def __repr__(self) -> str:
        return "Lebesgue measure"
