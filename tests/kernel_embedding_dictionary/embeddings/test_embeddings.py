# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_expquad_lebesgue_1d_1():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_1d_2():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_2d_1():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_2d_2():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
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
        [0.24216467221751733, 0.2486510713545889],
        [0.24630363249007906, 0.2527071991123484],
        [0.24701635519133064, 0.2534341832086769],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_2d_1():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_2d_1()
    mean_intervals = [
        [0.8196870021193354, 0.8219827266254676],
        [0.9207579814859344, 0.9216710074102283],
        [0.8193363233387485, 0.8216278133280472],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_2d_2():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_2d_2()
    mean_intervals = [
        [0.022810691847351294, 0.025168802301165173],
        [0.021634162578259314, 0.02390160606875473],
        [0.020470357518382998, 0.022648072052878],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_expquad_lebesgue_1d_1",
    "config_expquad_lebesgue_1d_2",
    "config_expquad_lebesgue_2d_1",
    "config_expquad_lebesgue_2d_2",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_uni_mean(fixture_name, request):
    kn, mn, ck, cm, x_eval, mean_intervals = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    for i in range(x_eval.shape[0]):
        print(i)
        print(res)
        assert mean_intervals[i][0] < res[i] < mean_intervals[i][1]
