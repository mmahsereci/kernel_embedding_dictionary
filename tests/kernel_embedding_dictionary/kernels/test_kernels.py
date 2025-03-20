# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary.kernels import (
    ExpQuadKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    Matern72Kernel,
    Wendland0Kernel,
    Wendland2Kernel,
    Wendland4Kernel,
)


@pytest.fixture()
def expquad():
    c = {"ndim": 2}
    return ExpQuadKernel(c)


@pytest.fixture()
def matern12():
    c = {"ndim": 2}
    return Matern12Kernel(c)


@pytest.fixture()
def matern32():
    c = {"ndim": 2}
    return Matern32Kernel(c)


@pytest.fixture()
def matern52():
    c = {"ndim": 2}
    return Matern52Kernel(c)


@pytest.fixture()
def matern72():
    c = {"ndim": 2}
    return Matern72Kernel(c)


@pytest.fixture()
def wendland0():
    c = {"ndim": 2}
    return Wendland0Kernel(c)


@pytest.fixture()
def wendland2():
    c = {"ndim": 2}
    return Wendland2Kernel(c)


@pytest.fixture()
def wendland4():
    c = {"ndim": 2}
    return Wendland4Kernel(c)


# for a new kernel: add a fixture and its name to the list
kernel_list = ["expquad", "matern12", "matern32", "matern52", "matern72", "wendland0", "wendland2"]


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_names(kernel_name, request):
    k = request.getfixturevalue(kernel_name)
    print(k)
    assert k.name == kernel_name


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_param_dict_from_dim(kernel_name, request):
    k = request.getfixturevalue(kernel_name)
    assert isinstance(k.get_param_dict_from_dim(0), dict)
    assert isinstance(k.get_param_dict_from_dim(1), dict)


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_evaluate_shapes(kernel_name, request):
    k = request.getfixturevalue(kernel_name)

    x1 = np.array([[0, 1], [2, 3], [4, 5]])  # 3x2
    x2 = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])  # 4x2
    res = k.evaluate(x1, x2)
    assert res.shape == (3, 4)

    # x1 has one entry only

    x1 = np.array([[0, 1]])  # 1x2
    x2 = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])  # 4x2
    res = k.evaluate(x1, x2)
    assert res.shape == (1, 4)

    # x2 has one entry only
    x1 = np.array([[0, 1], [2, 3], [4, 5]])  # 3x2
    x2 = np.array([[0, 1]])  # 1x2
    res = k.evaluate(x1, x2)
    assert res.shape == (3, 1)

    # x1 and x2 have one entry only
    x1 = np.array([[0, 1]])  # 1x2
    x2 = np.array([[0, 1]])  # 1x2
    res = k.evaluate(x1, x2)
    assert res.shape == (1, 1)


@pytest.mark.parametrize("kernel_name", kernel_list)
def test_kernel_uni_raises(kernel_name, request):
    k = request.getfixturevalue(kernel_name)

    x = np.array([[0, 1], [2, 3], [4, 5]])  # 3x2

    # wrong shape 1
    wrong_x = np.array([0, 1, 1])  # (3,)
    with pytest.raises(ValueError):
        k.evaluate(wrong_x, x)
    with pytest.raises(ValueError):
        k.evaluate(x, wrong_x)

    # wrong shape 2
    wrong_x = np.array([[[0, 1, 1]]])  # (1,1,3)
    with pytest.raises(ValueError):
        k.evaluate(wrong_x, x)
    with pytest.raises(ValueError):
        k.evaluate(x, wrong_x)

    # dimension mismatch
    wrong_x = np.array([[0, 1, 1], [2, 3, 1], [4, 5, 1]])  # (3, 2)
    with pytest.raises(ValueError):
        k.evaluate(wrong_x, x)
    with pytest.raises(ValueError):
        k.evaluate(x, wrong_x)
