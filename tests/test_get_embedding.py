# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding
from kernel_embedding_dictionary.kernels import (
    ExpQuadKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    Matern72Kernel,
    MaternKernel,
)
from kernel_embedding_dictionary.measures import GaussianMeasure, LebesgueMeasure

# add new combination to list to include it in tests


@pytest.mark.parametrize(
    "embedding",
    [
        ("expquad", ExpQuadKernel, "lebesgue", LebesgueMeasure),
        ("expquad", ExpQuadKernel, "gaussian", GaussianMeasure),
        ("matern", MaternKernel, "lebesgue", LebesgueMeasure),
        ("matern12", Matern12Kernel, "lebesgue", LebesgueMeasure),
        ("matern12", Matern12Kernel, "gaussian", GaussianMeasure),
        ("matern32", Matern32Kernel, "lebesgue", LebesgueMeasure),
        ("matern32", Matern32Kernel, "gaussian", GaussianMeasure),
        ("matern52", Matern52Kernel, "lebesgue", LebesgueMeasure),
        ("matern72", Matern72Kernel, "lebesgue", LebesgueMeasure),
    ],
)
def test_get_embedding_returns_correct_types(embedding):
    k_name, k_type, m_name, m_type = embedding
    ke = get_embedding(k_name, m_name)

    assert isinstance(ke._kernel, k_type)
    assert isinstance(ke._measure, m_type)


def test_get_embedding_raises():

    wrong_kernel_name = "unknown_kernel"
    wrong_measure_name = "unknown_measure"

    with pytest.raises(ValueError):
        get_embedding(wrong_kernel_name, "lebesgue")

    with pytest.raises(ValueError):
        get_embedding("expquad", wrong_measure_name)
