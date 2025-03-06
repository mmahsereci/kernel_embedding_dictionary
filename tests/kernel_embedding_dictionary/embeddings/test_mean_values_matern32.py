# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_matern32_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern32", "lebesgue", ck, cm, x


def get_config_matern32_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "matern32", "lebesgue", ck, cm, x


def get_config_matern32_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "matern32", "lebesgue", ck, cm, x


def get_config_matern32_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "matern32", "lebesgue", ck, cm, x


@pytest.fixture()
def config_matern32_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_matern32_lebesgue_1d_standard()
    mean_intervals = [
        [0.820737439660054, 0.8235787737501523],
        [0.9165847967937368, 0.9178436037126141],
        [0.8202419925658001, 0.823085176120207],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern32_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_matern32_lebesgue_1d_values()
    mean_intervals = [
        [0.21610115375550004, 0.22192902576732387],
        [0.2245228691439763, 0.23024105683269191],
        [0.22718952315756094, 0.23291587988640547],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern32_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_matern32_lebesgue_2d_standard()
    mean_intervals = [
        [0.7086199538034126, 0.7118086768569885],
        [0.8406790527858831, 0.8423163386137075],
        [0.7065961505859667, 0.7097730873749847],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern32_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_matern32_lebesgue_2d_values()
    mean_intervals = [
        [0.019368249545634937, 0.021370240498359103],
        [0.018925151962327844, 0.02088312983941415],
        [0.01780050462158635, 0.019638493169358246],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_matern32_lebesgue_1d_standard",
    "config_matern32_lebesgue_1d_values",
    "config_matern32_lebesgue_2d_standard",
    "config_matern32_lebesgue_2d_values",
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
