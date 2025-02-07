# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.measures import LebesgueMeasure


@pytest.fixture()
def lebesgue():
    c = {"ndim": 2}
    return LebesgueMeasure(c)


# for a new kernel: add a fixture and its name to the list
measure_list = ["lebesgue"]


@pytest.mark.parametrize("measure_name", measure_list)
def test_measure_names(measure_name, request):
    m = request.getfixturevalue(measure_name)
    assert m.name == measure_name


@pytest.mark.parametrize("measure_name", measure_list)
def test_measure_param_dict_from_dim(measure_name, request):
    m = request.getfixturevalue(measure_name)
    assert isinstance(m.get_param_dict_from_dim(0), dict)
    assert isinstance(m.get_param_dict_from_dim(1), dict)
