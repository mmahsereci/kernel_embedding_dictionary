# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.embeddings import KernelEmbedding
from kernel_embedding_dictionary.kernels import ExpQuadKernel
from kernel_embedding_dictionary.measures import GaussianMeasure


def test_kernel_embedding_raises():

    # dimension mismatch
    k_1d = ExpQuadKernel({"ndim": 1})
    m_2d = GaussianMeasure({"ndim": 2})
    with pytest.raises(ValueError):
        KernelEmbedding(k_1d, m_2d)
