# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import List, Optional

import numpy as np

from .measure import ProductMeasure, UnivariateMeasure


class GaussianMeasureUni(UnivariateMeasure):
    def __init__(self, mean: float, variance: float):

        if variance <= 0.0:
            raise ValueError(f"Variance ({variance}) must be postive.")

        self.mean = mean
        self.variance = variance

    def get_param_dict(self) -> dict:
        return {"mean": self.mean, "variance": self.variance}

    def sample(self, num_points: int) -> np.ndarray:
        return np.random.randn(num_points) * np.sqrt(self.variance) + self.mean


class GaussianMeasure(ProductMeasure):
    def __init__(self, config: Optional[dict] = None):
        """

        :param config:
        """
        # this will yield the standard Gaussian in 1D
        if config is None:
            config = {}

        # dimensionality and bounds
        variances = config.get("variances", None)
        means = config.get("means", None)
        ndim = config.get("ndim", None)

        # none given
        if not (means or variances or ndim):
            ndim = 1
            means = ndim * [0.0]
            variances = ndim * [1.0]
        # all given
        elif means and variances and ndim:
            if len(means) != len(variances) != ndim:
                raise ValueError("means, variances and ndim must match.")
        # only ndim given
        elif ndim and not (means or variances):
            means = ndim * [0.0]
            variances = ndim * [1.0]
        # only means given
        elif means and not (ndim or variances):
            variances = len(means) * [1.0]
        # only variances given
        elif variances and not (ndim or means):
            means = len(variances) * [0.0]
        # only variances given
        elif variances and not (ndim or means):
            means = len(variances) * [0.0]
        # ndim and variances given
        elif (variances and ndim) and not means:
            if len(variances) != ndim:
                raise ValueError("means, variances and ndim must match.")
            means = ndim * [0.0]
        # ndim and means given
        elif (means and ndim) and not variances:
            if len(means) != ndim:
                raise ValueError("means, variances and ndim must match.")
            variances = ndim * [1.0]
        # means and variances given
        else:
            if len(means) != len(variances):
                raise ValueError("means, variances and ndim must match.")

        measures = [GaussianMeasureUni(mean=m, variance=v) for (m, v) in zip(means, variances)]
        super().__init__(name="gaussian", measure_list=measures)  # sets name, ndim and measure list

    @property
    def means(self) -> List[float]:
        return [m.mean for m in self._measures]

    @property
    def variances(self) -> List[float]:
        return [m.variance for m in self._measures]

    def __str__(self) -> str:
        return (
            f"Gaussian measure \n"
            f"dimensionality: {self.ndim} \n"
            f"means: {self.means}\n"
            f"variances: {self.variances}"
        )

    def __repr__(self) -> str:
        return "Gaussian measure"
