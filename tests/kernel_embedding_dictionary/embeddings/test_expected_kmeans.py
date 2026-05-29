# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

from kernel_embedding_dictionary._get_embedding import get_embedding

from scipy.integrate import quad


def get_config_expquad_lebesgue_1d_standard():
    ck = {"ndim": 1}
    cm = {"ndim": 1}
    return "expquad", "lebesgue", ck, cm


def get_config_expquad_lebesgue_1d_values():
    ck = {"ndim": 1, "lengthscales": [0.3]}
    cm = {"ndim": 1, "bounds": [(-0.5, 2.5)], "normalize": True}  # test only works for normalized measures
    return "expquad", "lebesgue", ck, cm


@pytest.fixture()
def config_expquad_lebesgue_1d_standard():
    kn, mn, ck, cm = get_config_expquad_lebesgue_1d_standard()
    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    def ke_mean_scalar(x):
        return ke.mean(np.asarray(x).reshape(1, -1))

    ekm, int_err = quad(ke_mean_scalar, 0, 1)
    return kn, mn, ck, cm, ekm, int_err


@pytest.fixture()
def config_expquad_lebesgue_1d_values():
    kn, mn, ck, cm = get_config_expquad_lebesgue_1d_values()
    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)

    bounds = (cm["bounds"][0][0], cm["bounds"][0][1])

    def ke_mean_scalar(x):
        return ke.mean(np.asarray(x).reshape(1, -1)) / (bounds[1] - bounds[0])

    ekm, int_err = quad(ke_mean_scalar, *bounds)
    return kn, mn, ck, cm, ekm, int_err

fixture_list = [
    "config_expquad_lebesgue_1d_standard",
    "config_expquad_lebesgue_1d_values",
]

@pytest.mark.parametrize("fixture_name", fixture_list)
def test_expquad_lebesgue_mean_func_1d(fixture_name, request):
    # Test cases for the expected mean function
    kn, mn, ck, cm, num_ekm, int_err = request.getfixturevalue(fixture_name)
    ke = get_embedding(kernel_name=kn, measure_name=mn, kernel_config=ck, measure_config=cm)
    assert num_ekm == pytest.approx(ke.variance(), rel=int_err)