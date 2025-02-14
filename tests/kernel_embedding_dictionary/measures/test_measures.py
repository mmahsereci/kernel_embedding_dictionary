# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.measures import GaussianMeasure, LebesgueMeasure

# common tests for all measures go into this file. Add a fixture to add a measre to the tests
# - measure name
# - sample shapes
#
# test that are measure specific go to the specific test file
# - default param values
# - param values
# - raises

NDIM = 2


@pytest.fixture()
def lebesgue():
    c = {"ndim": NDIM, "bounds": [(1.0, 2.5), (0.0, 1.0)], "normalize": True}
    return LebesgueMeasure(c)


@pytest.fixture()
def gaussian():
    c = {"ndim": NDIM, "means": [-0.5, 0.3], "variances": [0.2, 1.4]}
    return GaussianMeasure(c)


# for a new kernel: add a fixture and its name to the list
measure_list = ["lebesgue", "gaussian"]


@pytest.mark.parametrize("measure_name", measure_list)
def test_measure_names(measure_name, request):
    m = request.getfixturevalue(measure_name)
    assert m.name == measure_name


@pytest.mark.parametrize("measure_name", measure_list)
def test_measure_param_dict_from_dim(measure_name, request):
    m = request.getfixturevalue(measure_name)
    for i in range(NDIM):
        assert isinstance(m.get_param_dict_from_dim(i), dict)


@pytest.mark.parametrize("measure_name", measure_list)
def test_measure_sample_shapes(measure_name, request):
    m = request.getfixturevalue(measure_name)
    num_points = 5
    res = m.sample(num_points)
    assert res.shape == (num_points, NDIM)
