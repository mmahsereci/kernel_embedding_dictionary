# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from kernel_embedding_dictionary.embeddings import KernelEmbedding
from kernel_embedding_dictionary import get_embedding


@pytest.fixture()
def config_expquad_lebesgue_uni_1():
    ck = {"ndim": 1, "lengthscale": [1.0]}
    cm = {"ndim": 1, "bounds": [(0.0, 1.0)]}
    x = np.array([[1, 0, 1]])
    mean_intervals = [[0, 1], [], []]
    return "expquad", "lebesgue", ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_uni_2():
    ck = {"ndim": 1, "lengthscale": [0.3]}
    cm = {"ndim": 1, "lengthscale": [(-0.5, 2.5)]}
    mean_interval = [0, 1]
    return "expquad", "lebesgue", ck, cm, mean_interval


fixture_list = [
    "config_expquad_lebesgue_uni_1",
    "config_expquad_lebesgue_uni_2",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_uni_mean(fixture_name, request):
    k_name, m_name, ck, cm, mean_intervals, x = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=k_name, measure_name=m_name, kernel_config=ck, measure_config=cm)

    for i, xi in enumerate(x):
        assert mean_intervals[i, 0] < res[i] < mean_intervals[i, 1]
