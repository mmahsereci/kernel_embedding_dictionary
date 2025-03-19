# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_wendland2_gaussian_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[-0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "wendland2", "gaussian", ck, cm, x


def get_config_wendland2_gaussian_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [0.0], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "wendland2", "gaussian", ck, cm, x


def get_config_wendland2_gaussian_1d_non_zero_mean_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [0.5], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "wendland2", "gaussian", ck, cm, x


def get_config_wendland2_gaussian_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "wendland2", "gaussian", ck, cm, x


def get_config_wendland2_gaussian_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.3, 2.5]}
    cm = {"ndim": 2, "means": [0.0, 0.0], "variances": [1.7, 0.4]}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "wendland2", "gaussian", ck, cm, x


@pytest.fixture()
def config_wendland2_gaussian_1d_standard():
    kn, mn, ck, cm, x = get_config_wendland2_gaussian_1d_standard()
    mean_intervals = [
        [0.30169843765303206, 0.30852702951002886],
        [0.269905875452005, 0.2765661979552871],
        [0.20885760116535623, 0.2150829652507345],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland2_gaussian_1d_values():
    kn, mn, ck, cm, x = get_config_wendland2_gaussian_1d_values()
    mean_intervals = [
        [0.07036767663696422, 0.07445019053138918],
        [0.06722253954392475, 0.07123174505949019],
        [0.02736613340065509, 0.03002747066251169],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland2_gaussian_2d_standard():
    kn, mn, ck, cm, x = get_config_wendland2_gaussian_2d_standard()
    mean_intervals = [
        [0.08963279954649817, 0.0934379778909343],
        [0.08113699318035601, 0.08479491903629026],
        [0.00020700907413322275, 0.00041760023615360856],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland2_gaussian_2d_values():
    kn, mn, ck, cm, x = get_config_wendland2_gaussian_2d_values()
    mean_intervals = [
        [0.054612552326695527, 0.057938474331984735],
        [0.05247147607213139, 0.05575066814357686],
        [5.556009312582308e-05, 0.00011257518996163838],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_wendland2_gaussian_1d_standard",
    "config_wendland2_gaussian_1d_values",
    "config_wendland2_gaussian_2d_standard",
    "config_wendland2_gaussian_2d_values",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_mean_values(fixture_name, request):
    kn, mn, ck, cm, x_eval, mean_intervals = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    print(res)
    for i in range(x_eval.shape[0]):
        assert mean_intervals[i][0] < res[i] < mean_intervals[i][1]


def test_get_embedding_raises():
    """Test that get_embedding raises ValueError for non-zero mean in Gaussian measure."""
    kn, mn, ck, cm, x = get_config_wendland2_gaussian_1d_non_zero_mean_values()

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    with pytest.raises(ValueError):
        res = ke.mean(x)
