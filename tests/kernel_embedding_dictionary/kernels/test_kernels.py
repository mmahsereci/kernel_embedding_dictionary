# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.kernels import ExpQuadKernel

# shapes of evaluate

@pytest.fixture()
def expquad():
    c = {"ndim": 2}
    return ExpQuadKernel(c)


# for a new kernel: add a fixture and its name to the list
kernel_list = ["expquad"]


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_names(kernel_name, request):
    k = request.getfixturevalue(kernel_name)
    assert k.name == kernel_name


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_param_dict_from_dim(kernel_name, request):
    k = request.getfixturevalue(kernel_name)
    assert isinstance(k.get_param_dict_from_dim(0), dict)
    assert isinstance(k.get_param_dict_from_dim(1), dict)
