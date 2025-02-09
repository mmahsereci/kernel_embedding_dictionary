# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from kernel_embedding_dictionary.embeddings import KernelEmbedding
from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_expquad_lebesgue_1d_1():
    ck = {"ndim": 1, "lengthscale": [1.0]}
    cm = {"ndim": 1, "bounds": [(0.0, 1.0)]}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_1d_2():
    ck = {"ndim": 1, "lengthscale": [0.03]}
    cm = {"ndim": 1, "lengthscale": [(-0.5, 2.5)]}
    x = np.array([[-0.3], [0.0], [1.5]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


@pytest.fixture()
def config_expquad_lebesgue_1d_1():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_1d_1()
    mean_intervals = [
        [0.8908900366778463, 0.8928485226983125],
        [0.9594153133777275, 0.9600861603772526],
        [0.8905492816317769, 0.8925099674274802],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_1d_2():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_1d_2()
    mean_intervals = [
        [0.7123375363149562, 0.7153457299975337],
        [0.8535104942181617, 0.8558150970311784],
        [0.60567942156233, 0.6088408610742959]
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_expquad_lebesgue_1d_1",
    "config_expquad_lebesgue_1d_2",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_uni_mean(fixture_name, request):
    kn, mn, ck, cm, x_eval, mean_intervals = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    for i in range(x_eval.shape[0]):
        assert mean_intervals[i][0] < res[i] < mean_intervals[i][1]
