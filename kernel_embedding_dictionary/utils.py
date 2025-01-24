# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np

from typing import Union


def scaled_vector_diff(
    x1: Union[np.ndarray, float], x2: Union[np.ndarray, float], scale: float
) -> Union[np.ndarray, float]:
    return (x1 - x2) / (scale * np.sqrt(2))
