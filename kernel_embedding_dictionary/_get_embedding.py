# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from typing import Optional

from .embeddings import KernelEmbedding
from .kernels import ExpQuadKernel
from .measures import LebesgueMeasure


# TODO: add return type
def get_embedding(
    kernel_name: str, measure_name: str, kernel_config: Optional[dict] = None, measure_config: Optional[dict] = None
):
    """Constructs the kernel embedding from the given configurations.

    :param kernel_name:
    :param measure_name:
    :param kernel_config:
    :param measure_config:
    :return:
    """

    # TODO get kernel and measure instance and create KernelEmbedding instance
    if kernel_name == "expquad":
        if measure_name == "lebesgue":
            return KernelEmbedding(ExpQuadKernel(kernel_config), LebesgueMeasure(measure_config))
