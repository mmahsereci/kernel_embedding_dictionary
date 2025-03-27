# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_matern52_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern52", "lebesgue", ck, cm, x


def get_config_matern52_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "matern52", "lebesgue", ck, cm, x


def get_config_matern52_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "matern52", "lebesgue", ck, cm, x


def get_config_matern52_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "matern52", "lebesgue", ck, cm, x


@pytest.fixture()
def config_matern52_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_matern52_lebesgue_1d_standard()
    mean_intervals = [
        [0.8505856727206377, 0.8531135261754442],
        [0.938034149740373, 0.9390260342881223],
        [0.8501473862694101, 0.8526775391422204],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern52_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_matern52_lebesgue_1d_values()
    mean_intervals = [
        [0.22474163643363446, 0.2308187296267544],
        [0.232426893319601, 0.23840274351595123],
        [0.2347520825010043, 0.24074104484570116],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern52_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_matern52_lebesgue_2d_standard()
    mean_intervals = [
        [0.7565210732612763, 0.7594132799160161],
        [0.8803462371107011, 0.8816657350683541],
        [0.7547079654510293, 0.7575922708885253],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern52_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_matern52_lebesgue_2d_values()
    mean_intervals = [
        [0.020511832593311546, 0.022645577391982953],
        [0.02012213899212245, 0.022215266659219225],
        [0.01910260170996778, 0.021083400822967894],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_matern52_lebesgue_1d_standard",
    "config_matern52_lebesgue_1d_values",
    "config_matern52_lebesgue_2d_standard",
    "config_matern52_lebesgue_2d_values",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_mean_values(fixture_name, request):
    kn, mn, ck, cm, x_eval, mean_intervals = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    for i in range(x_eval.shape[0]):
        print(i)
        print(res)
        assert mean_intervals[i][0] < res[i] < mean_intervals[i][1]
