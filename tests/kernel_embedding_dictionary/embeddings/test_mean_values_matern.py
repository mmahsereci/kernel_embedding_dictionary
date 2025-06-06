# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary._get_embedding import get_embedding


def get_config_matern_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    x = np.array([[0.1], [0.5], [0.9]])  # 3x1 must lie in domain
    return "matern", "lebesgue", ck, cm, x


def get_config_matern_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    x = np.array([[0.1], [0.5], [0.85]])  # 3x1 must lie in domain
    return "matern", "lebesgue", ck, cm, x


def get_config_matern_lebesgue_2d_standard():
    ck = {"ndim": 2}
    cm = {"ndim": 2}
    x = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.2]])  # 3x2 must lie in domain
    return "matern", "lebesgue", ck, cm, x


def get_config_matern_lebesgue_2d_values():
    ck = {"ndim": 2, "lengthscales": [0.03, 1.6]}
    cm = {"ndim": 2, "bounds": [(-0.5, 2.5), (-1.5, 0.1)], "normalize": True}  # test only works for normalized measures
    x = np.array([[-0.3, -1], [0.0, -0.2], [1.5, 0.0]])  # 3x2 must lie in domain
    return "matern", "lebesgue", ck, cm, x


@pytest.fixture()
def config_matern_lebesgue_1d_standard():
    kn, mn, ck, cm, x = get_config_matern_lebesgue_1d_standard()
    return kn, mn, ck, cm, x


@pytest.fixture()
def config_matern_lebesgue_1d_values():
    kn, mn, ck, cm, x = get_config_matern_lebesgue_1d_values()
    return kn, mn, ck, cm, x


@pytest.fixture()
def config_matern_lebesgue_2d_standard():
    kn, mn, ck, cm, x = get_config_matern_lebesgue_2d_standard()
    return kn, mn, ck, cm, x


@pytest.fixture()
def config_matern_lebesgue_2d_values():
    kn, mn, ck, cm, x = get_config_matern_lebesgue_2d_values()
    return kn, mn, ck, cm, x


fixture_list = [
    "config_matern_lebesgue_1d_standard",
    "config_matern_lebesgue_1d_values",
    "config_matern_lebesgue_2d_standard",
    "config_matern_lebesgue_2d_values",
]


@pytest.mark.parametrize("nu_set", [("12", 0.5), ("32", 1.5), ("52", 2.5), ("72", 3.5)])
@pytest.mark.parametrize("fixture_name", fixture_list)
def test_embedding_mean_values_closed_form(nu_set, fixture_name, request):
    kn, mn, ck, cm, x_eval = request.getfixturevalue(fixture_name)
    nu_suffix, nu = nu_set

    ck.update({"nu": nu})
    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)
    ke_explicit = get_embedding(kernel_name=kn + nu_suffix, measure_name=mn, kernel_config=ck, measure_config=cm)

    res = ke.mean(x_eval)
    res_explicit = ke_explicit.mean(x_eval)
    for i in range(x_eval.shape[0]):
        print(i, nu)
        print(res)
        print(res_explicit)
        assert res[i] == pytest.approx(res_explicit[i])

