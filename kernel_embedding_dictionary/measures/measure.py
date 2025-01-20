# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import abc
import numpy as np

class Measure(abc.ABC):

    # TODO: add types
    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    # TODO: is it good to handle these as abstract methods?
    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    # TODO: is it good to handle these as abstract methods?
    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
