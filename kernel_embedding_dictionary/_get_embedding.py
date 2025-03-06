# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Optional

from .embeddings import KernelEmbedding
from .kernels import ExpQuadKernel, Matern12Kernel, Matern32Kernel
from .measures import GaussianMeasure, LebesgueMeasure


def get_embedding(
    kernel_name: str, measure_name: str, kernel_config: Optional[dict] = None, measure_config: Optional[dict] = None
) -> KernelEmbedding:

    available_embeddings_dict = {
        "expquad-lebesgue": [ExpQuadKernel, LebesgueMeasure],
        "expquad-gaussian": [ExpQuadKernel, GaussianMeasure],
        "matern12-lebesgue": [Matern12Kernel, LebesgueMeasure],
        "matern32-lebesgue": [Matern32Kernel, LebesgueMeasure],
    }

    km = available_embeddings_dict.get(kernel_name + "-" + measure_name, None)

    if not km:
        raise ValueError(f"kernel embedding unknown.")

    return KernelEmbedding(km[0](kernel_config), km[1](measure_config))
