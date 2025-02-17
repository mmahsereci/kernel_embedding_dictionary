# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
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


def test_kernel_embedding_mean_raises():

    # dimension mismatch
    k_2d = ExpQuadKernel({"ndim": 2})
    m_2d = GaussianMeasure({"ndim": 2})
    ke_2d = KernelEmbedding(k_2d, m_2d)

    # x dimension wrong
    wrong_x = np.ones([5, 3])
    with pytest.raises(ValueError):
        ke_2d.mean(wrong_x)

    # x shape wrong (too large)
    wrong_x = np.ones([5, 2, 1])
    with pytest.raises(ValueError):
        ke_2d.mean(wrong_x)

    # x shape wrong (too small)
    k_1d = ExpQuadKernel({"ndim": 1})
    m_1d = GaussianMeasure({"ndim": 1})
    ke_1d = KernelEmbedding(k_1d, m_1d)
    wrong_x = np.ones(5)
    with pytest.raises(ValueError):
        ke_1d.mean(wrong_x)
