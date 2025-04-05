# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_matern12_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern12", "lebesgue", ck, cm, x


def get_config_matern12_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "matern12", "lebesgue", ck, cm, x


def get_config_matern12_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "matern12", "lebesgue", ck, cm, x


def get_config_matern12_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "matern12", "lebesgue", ck, cm, x


def get_config_matern12_gaussian_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[-0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern12", "gaussian", ck, cm, x


def get_config_matern12_gaussian_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [-0.6], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "matern12", "gaussian", ck, cm, x


def get_config_matern12_gaussian_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "matern12", "gaussian", ck, cm, x


def get_config_matern12_gaussian_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.3, 2.5]}
    cm = {"ndim": 2, "means": [-0.6, 0.2], "variances": [1.7, 0.4]}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "matern12", "gaussian", ck, cm, x


@pytest.fixture()
def config_matern12_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_matern12_lebesgue_1d_standard()
    mean_intervals = [
        [0.6872641697546467, 0.6907605280076654],
        [0.7856108222926309, 0.787763695452152],
        [0.6865542841473361, 0.6900470198238321],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_matern12_lebesgue_1d_values()
    mean_intervals = [
        [0.18237014049054617, 0.18717116158450436],
        [0.19218283330834213, 0.19686267334623397],
        [0.19570895363339136, 0.20038069134443104],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_matern12_lebesgue_2d_standard()
    mean_intervals = [
        [0.5037877480728302, 0.5072147448784917],
        [0.6179191410665993, 0.6203270954295531],
        [0.5014550324231958, 0.5048555311192656],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_matern12_lebesgue_2d_values()
    mean_intervals = [
        [0.014375999230232691, 0.015819468491738948],
        [0.013949817190146712, 0.015354090184312602],
        [0.012808746587904122, 0.014100149774921649],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_gaussian_1d_standard():
    kn, mn, ck, cm, x = get_config_matern12_gaussian_1d_standard()
    mean_intervals = [
        [0.5207811600570372, 0.5255327905232955],
        [0.4890161476742793, 0.4938876448313159],
        [0.4236371559990272, 0.42861101565670334],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_gaussian_1d_values():
    kn, mn, ck, cm, x = get_config_matern12_gaussian_1d_values()
    mean_intervals = [
        [0.1628671969443623, 0.1674440150652347],
        [0.12470600891623922, 0.12885476436394938],
        [0.036235870456771374, 0.038630587793021684],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_gaussian_2d_standard():
    kn, mn, ck, cm, x = get_config_matern12_gaussian_2d_standard()
    mean_intervals = [
        [0.26891306118311886, 0.27262100452815163],
        [0.2544125717935101, 0.25807818550902833],
        [0.022930787220573984, 0.02363649030211013],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_matern12_gaussian_2d_values():
    kn, mn, ck, cm, x = get_config_matern12_gaussian_2d_values()
    mean_intervals = [
        [0.13431019223369556, 0.13813740605960861],
        [0.1030581490590364, 0.10654141580994095],
        [0.027106081266712233, 0.028218637869088944],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_matern12_lebesgue_1d_standard",
    "config_matern12_lebesgue_1d_values",
    "config_matern12_lebesgue_2d_standard",
    "config_matern12_lebesgue_2d_values",
    "config_matern12_gaussian_1d_standard",
    "config_matern12_gaussian_1d_values",
    "config_matern12_gaussian_2d_standard",
    "config_matern12_gaussian_2d_values",
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
