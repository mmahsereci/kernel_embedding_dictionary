# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_wendland0_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "wendland0", "lebesgue", ck, cm, x


def get_config_wendland0_lebesgue_1d_values_case1to3():
    """Tests three conditions of the Wendland0 kernel mean embedding with Lebesgue measure in 1D.

    These settings cover three of four cases in the integrated Wendland kernel:
    1. (ub >= (x + ell)) & ((lb + ell) < x)
    2. (ub >= (x + ell)) & ((lb + ell) >= x)
    3. (ub < (x + ell)) & ((lb + ell) < x)
    """
    ck = {"ndim": 1, "lengthscales": [0.8]}
    cm = {"ndim": 1, "bounds": [(-0.3, 1.4)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.55], [0.8]])  # 3x1 must lie in domain
    return "wendland0", "lebesgue", ck, cm, x


def get_config_wendland0_lebesgue_1d_values_case4():
    """Tests the fourth condition (not 1, 2 or 3) of the Wendland0 kernel mean embedding with Lebesgue measure in 1D."""
    ck = {"ndim": 1, "lengthscales": [0.9]}
    cm = {"ndim": 1, "bounds": [(-0.3, 1.4)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.55]])  # 1x1 must lie in domain
    return "wendland0", "lebesgue", ck, cm, x


def get_config_wendland0_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "wendland0", "lebesgue", ck, cm, x


def get_config_wendland0_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "wendland0", "lebesgue", ck, cm, x


def get_config_wendland0_gaussian_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[-0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "wendland0", "gaussian", ck, cm, x


def get_config_wendland0_gaussian_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [0.0], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "wendland0", "gaussian", ck, cm, x


def get_config_wendland0_gaussian_1d_non_zero_mean_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "means": [0.5], "variances": [1.7]}
    x = np.array([[-0.1], [0.5], [1.8]])  # 3x1 must lie in domain
    return "wendland0", "gaussian", ck, cm, x


def get_config_wendland0_gaussian_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "wendland0", "gaussian", ck, cm, x


def get_config_wendland0_gaussian_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.3, 2.5]}
    cm = {"ndim": 2, "means": [0.0, 0.0], "variances": [1.7, 0.4]}
    x = np.array([[-0.1, 0.2], [0.5, 0.0], [0.9, 3.4]])  # 3x1 must lie in domain
    return "wendland0", "gaussian", ck, cm, x


@pytest.fixture()
def config_wendland0_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_wendland0_lebesgue_1d_standard()
    mean_intervals = [
        [0.5878973312675195, 0.5931092086395331],
        [0.7482878315940619, 0.7510289352268248],
        [0.5868764192008101, 0.5920869926334976],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_wendland0_lebesgue_1d_values_case1to3()
    mean_intervals = [
        [0.40545133617900014, 0.41214700860517123],
        [0.4676560436265604, 0.4734209224151989],
        [0.454937583642213, 0.4610374882291332],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_lebesgue_1d_values_case4():
    kn, mn, ck, cm, x = get_config_wendland0_lebesgue_1d_values_case4()
    mean_intervals = [
        [0.5246965759851641, 0.5298613416032558],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_wendland0_lebesgue_2d_standard()
    mean_intervals = [
        [0.3892513090856746, 0.3938017460803801],
        [0.560848606251837, 0.5637831612028716],
        [0.385896693576288, 0.3904184623535142],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_wendland0_lebesgue_2d_values()
    mean_intervals = [
        [0.0064726628689735005, 0.007615612154027926],
        [0.006166576171899892, 0.007256721171136064],
        [0.0050902145657878515, 0.006047125061588528],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_gaussian_1d_standard():
    kn, mn, ck, cm, x = get_config_wendland0_gaussian_1d_standard()
    mean_intervals = [
        [0.364076904917011, 0.37057302342176984],
        [0.32999013949842987, 0.33648368540327595],
        [0.25948222921309627, 0.26570959614888057],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_gaussian_1d_values():
    kn, mn, ck, cm, x = get_config_wendland0_gaussian_1d_values()
    mean_intervals = [
        [0.08877430837032896, 0.09311611680644455],
        [0.08389713857769604, 0.0881535630583076],
        [0.03466435433160734, 0.03752774936720384],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_gaussian_2d_standard():
    kn, mn, ck, cm, x = get_config_wendland0_gaussian_2d_standard()
    mean_intervals = [
        [0.13027047170793685, 0.13429803606094606],
        [0.11988858934822494, 0.12379426502963284],
        [0.00050360469939905, 0.0007823685535048706],
    ]
    return kn, mn, ck, cm, x, mean_intervals


@pytest.fixture()
def config_wendland0_gaussian_2d_values():
    kn, mn, ck, cm, x = get_config_wendland0_gaussian_2d_values()
    mean_intervals = [
        [0.06977894538796134, 0.07328018552496801],
        [0.06603235598510478, 0.06946551916369643],
        [0.0005390654902389064, 0.0007164047856198059],
    ]
    return kn, mn, ck, cm, x, mean_intervals


fixture_list = [
    "config_wendland0_lebesgue_1d_standard",
    "config_wendland0_lebesgue_1d_values",
    "config_wendland0_lebesgue_1d_values_case4",
    "config_wendland0_lebesgue_2d_standard",
    "config_wendland0_lebesgue_2d_values",
    "config_wendland0_gaussian_1d_standard",
    "config_wendland0_gaussian_1d_values",
    "config_wendland0_gaussian_2d_standard",
    "config_wendland0_gaussian_2d_values",
]


@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_mean_values(fixture_name, request):
    kn, mn, ck, cm, x_eval, mean_intervals = request.getfixturevalue(fixture_name)

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    print(res)
    for i in range(x_eval.shape[0]):
        assert mean_intervals[i][0] < res[i] < mean_intervals[i][1]


def test_kernel_mean_raises():
    """Test that the kernel mean raises ValueError for non-zero mean in Gaussian measure."""
    kn, mn, ck, cm, x = get_config_wendland0_gaussian_1d_non_zero_mean_values()

    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    with pytest.raises(ValueError):
        res = ke.mean(x)
