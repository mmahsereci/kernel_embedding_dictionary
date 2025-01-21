# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np


def scaled_vector_diff(x1: np.ndarray, x2: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (x1 - x2) / (scales * np.sqrt(2))
