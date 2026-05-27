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

    @property
    def param_dict(self) -> dict:
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

        means = config.get("means", None)
        variances = config.get("variances", None)
        ndim = config.get("ndim", None)

        # validate all provided values against each other
        if means is not None and variances is not None and len(means) != len(variances):
            raise ValueError(f"len(means) ({len(means)}) and len(variances) ({len(variances)}) must match.")
        if means is not None and ndim is not None and len(means) != ndim:
            raise ValueError(f"len(means) ({len(means)}) and ndim ({ndim}) must match.")
        if variances is not None and ndim is not None and len(variances) != ndim:
            raise ValueError(f"len(variances) ({len(variances)}) and ndim ({ndim}) must match.")

        # infer ndim from first available source, default to 1
        if ndim is None:
            if means is not None:
                ndim = len(means)
            elif variances is not None:
                ndim = len(variances)
            else:
                ndim = 1

        # fill missing means/variances with defaults
        if means is None:
            means = ndim * [0.0]
        if variances is None:
            variances = ndim * [1.0]

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
