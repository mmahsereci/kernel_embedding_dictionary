# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_matern72_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern72", "lebesgue", ck, cm, x


def get_config_matern72_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "matern72", "lebesgue", ck, cm, x


def get_config_matern72_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "matern72", "lebesgue", ck, cm, x


def get_config_matern72_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "matern72", "lebesgue", ck, cm, x


@pytest.fixture()
def config_matern72_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_matern72_lebesgue_1d_standard()
    mean_intervals = [
        [0.8630353679348793, 0.8654060493414811],
        [0.9455510024162597, 0.9464358404571365],
        [0.8626240615359427, 0.8649971043548336],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern72_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_matern72_lebesgue_1d_values()
    mean_intervals = [
        [0.228693351775723, 0.2348810858204736],
        [0.2359541984245358, 0.24204592463298918],
        [0.23810597313303009, 0.244212989052194],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern72_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_matern72_lebesgue_2d_standard()
    mean_intervals = [
        [0.7765184993250671, 0.7792501931865399],
        [0.8944578460823785, 0.8956441178283358],
        [0.7748128634279339, 0.7775380349293453],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern72_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_matern72_lebesgue_2d_values()
    mean_intervals = [
        [0.020996020661025475, 0.023186406418042885],
        [0.02063854873766817, 0.022790665522468466],
        [0.019680992244065304, 0.021725530886956758],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_matern72_lebesgue_1d_standard",
    "config_matern72_lebesgue_1d_values",
    "config_matern72_lebesgue_2d_standard",
    "config_matern72_lebesgue_2d_values",
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
