# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc

import numpy as np


class Kernel(abc.ABC):
    @abc.abstractmethod
    def k(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass
