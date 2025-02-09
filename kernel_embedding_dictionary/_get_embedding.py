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
    available_embeddings_dict = {"expquad-lebesgue": [ExpQuadKernel, LebesgueMeasure]}

    km = available_embeddings_dict.get(kernel_name + "-" + measure_name, None)

    if not km:
        raise ValueError(f"kernel embedding unknown.")

    return KernelEmbedding(km[0](kernel_config), km[1](measure_config))
