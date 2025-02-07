# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc
from typing import List


class UnivariateMeasure(abc.ABC):
    @abc.abstractmethod
    def get_param_dict(self) -> dict:
        pass


class ProductMeasure(abc.ABC):
    def __init__(self, name: str, measure_list: List[UnivariateMeasure]):
        self.name = name
        self._measures = measure_list

    def get_param_dict_from_dim(self, dim: int) -> dict:
        return self._measures[dim].get_param_dict()

    @property
    def ndim(self):
        return len(self._measures)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass
