# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc
import numpy as np


class Measure(abc.ABC):
    @abc.abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
