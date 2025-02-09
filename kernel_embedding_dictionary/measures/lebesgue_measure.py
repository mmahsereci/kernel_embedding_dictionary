# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Optional

import numpy as np

from .measure import ProductMeasure, UnivariateMeasure


class LebesgueMeasureUni(UnivariateMeasure):
    def __init__(self, lb: float, ub: float, normalize: bool):

        if lb >= ub:
            raise ValueError(f"upper bound ({ub})  must be larger than lower bound ({lb}).")

        self.lb = lb
        self.ub = ub

        density = 1.0
        if normalize:
            density = 1.0 / (ub - lb)
        self.normalize = normalize
        self.density = density

    def get_param_dict(self) -> dict:
        return {"lb": self.lb, "ub": self.ub, "density": self.density}

    def sample(self, num_points: int) -> np.ndarray:
        return np.random.rand(num_points) * (self.ub - self.lb) + self.lb


class LebesgueMeasure(ProductMeasure):
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

        if ndim is not None:
            if (ndim > 1) and (len(bounds) == 1):
                bounds = ndim * [bounds[0]]
            else:
                if ndim != len(bounds):
                    raise ValueError(f"ndim ({ndim}) and dimensionality fo bounds ({len(bounds)}) does not match.")

        # normalization and density
        normalize = config.get("normalize", False)
        self.normalize = normalize

        measures = [LebesgueMeasureUni(lb=lbi, ub=ubi, normalize=normalize) for (lbi, ubi) in bounds]
        super().__init__(name="lebesgue", measure_list=measures)  # sets name, ndim and measure list

    @property
    def lb(self):
        return [m.lb for m in self._measures]

    @property
    def ub(self):
        return [m.ub for m in self._measures]

    @property
    def bounds(self):
        return [(lb, ub) for lb, ub in zip(self.lb, self.ub)]

    def __str__(self) -> str:
        return (
            f"Lebesgue measure \n"
            f"dimensionality: {self.ndim} \n"
            f"normalized: {self.normalize}\n"
            f"bounds: {self.bounds}"
        )

    def __repr__(self) -> str:
        return "Lebesgue measure"
