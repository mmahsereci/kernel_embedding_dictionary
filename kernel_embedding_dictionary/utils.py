# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Union

import numpy as np


def scaled_diff(x1: Union[np.ndarray, float], x2: Union[np.ndarray, float], scale: float) -> Union[np.ndarray, float]:
    return (x1 - x2) / (scale * np.sqrt(2))
