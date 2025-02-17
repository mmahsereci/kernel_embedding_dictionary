# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_expquad_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "expquad", "lebesgue", ck, cm, x


def get_config_expquad_gaussian_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[-0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "expquad", "gaussian", ck, cm, x


def get_config_expquad_gaussian_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [-0.6], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "expquad", "gaussian", ck, cm, x


def get_config_expquad_gaussian_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "expquad", "gaussian", ck, cm, x


def get_config_expquad_gaussian_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.3, 2.5]}
    cm = {"ndim": 2, "means": [-0.6, 0.2], "variances": [1.7, 0.4]}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "expquad", "gaussian", ck, cm, x


@pytest.fixture()
def config_expquad_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_1d_standard()
    mean_intervals = [
        [0.8908900366778463, 0.8928485226983125],
        [0.9594153133777275, 0.9600861603772526],
        [0.8905492816317769, 0.8925099674274802],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_1d_values()
    mean_intervals = [
        [0.24216467221751733, 0.2486510713545889],
        [0.24630363249007906, 0.2527071991123484],
        [0.24701635519133064, 0.2534341832086769],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_2d_standard()
    mean_intervals = [
        [0.8196870021193354, 0.8219827266254676],
        [0.9207579814859344, 0.9216710074102283],
        [0.8193363233387485, 0.8216278133280472],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_expquad_lebesgue_2d_values()
    mean_intervals = [
        [0.022810691847351294, 0.025168802301165173],
        [0.021634162578259314, 0.02390160606875473],
        [0.020470357518382998, 0.022648072052878],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_gaussian_1d_standard():
    kn, mn, ck, cm, x = get_config_expquad_gaussian_1d_standard()
    mean_intervals = [
        [0.7042191433836388, 0.7095055195369598],
        [0.6632773727386115, 0.6689566602022193],
        [0.5758292281125416, 0.5820348101128473],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_gaussian_1d_values():
    kn, mn, ck, cm, x = get_config_expquad_gaussian_1d_values()
    mean_intervals = [
        [0.2080170595148499, 0.21422075691500153],
        [0.15766518237515253, 0.1632858613468225],
        [0.04309127377798668, 0.04630799779808741],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_gaussian_2d_standard():
    kn, mn, ck, cm, x = get_config_expquad_gaussian_2d_standard()
    mean_intervals = [
        [0.49247937555629934, 0.49797384250995963],
        [0.46835494267561445, 0.47391460236015087],
        [0.022199051259112652, 0.02353330407976511],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_expquad_gaussian_2d_values():
    kn, mn, ck, cm, x = get_config_expquad_gaussian_2d_values()
    mean_intervals = [
        [0.1985936949911137, 0.20457554670766034],
        [0.14978740424189563, 0.15519309004892373],
        [0.052155457802633434, 0.0545233118212778],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_expquad_lebesgue_1d_standard",
    "config_expquad_lebesgue_1d_values",
    "config_expquad_lebesgue_2d_standard",
    "config_expquad_lebesgue_2d_values",
    "config_expquad_gaussian_1d_standard",
    "config_expquad_gaussian_1d_values",
    "config_expquad_gaussian_2d_standard",
    "config_expquad_gaussian_2d_values",
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
