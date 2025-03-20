# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary.kernels import (
    ExpQuadKernelUni,
    Matern12KernelUni,
    Matern32KernelUni,
    Matern52KernelUni,
    Matern72KernelUni,
)


@pytest.fixture()
def expquad_uni():
    return ExpQuadKernelUni(ell=1.0)


@pytest.fixture()
def matern12_uni():
    return Matern12KernelUni(ell=1.0)


@pytest.fixture()
def matern32_uni():
    return Matern32KernelUni(ell=1.0)


@pytest.fixture()
def matern52_uni():
    return Matern52KernelUni(ell=1.0)


@pytest.fixture()
def matern72_uni():
    return Matern72KernelUni(ell=1.0)


# for a new univariate kernel: add a fixture and its name to the list
kernel_uni_list = ["expquad_uni", "matern12_uni", "matern32_uni", "matern52_uni"]


@pytest.mark.parametrize("kernel_uni_name", kernel_uni_list)
def test_kernel_uni_evaluate_shapes(kernel_uni_name, request):
    k = request.getfixturevalue(kernel_uni_name)

    x1 = np.array([1.0, 2.0, 1.5])
    x2 = np.array([0.1, 0.5])

    res = k.evaluate(x1, x2)
    assert res.shape == (3, 2)

    # x1 has one entry only
    x1 = np.array([1.0])
    x2 = np.array([0.1, 0.5])

    res = k.evaluate(x1, x2)
    assert res.shape == (1, 2)

    # x2 has one entry only
    x1 = np.array([1.0, 2.0, 1.5])
    x2 = np.array([0.1])

    res = k.evaluate(x1, x2)
    assert res.shape == (3, 1)

    # x1 ad x2 have one entry only
    x1 = np.array([1.0])
    x2 = np.array([0.1])

    res = k.evaluate(x1, x2)
    assert res.shape == (1, 1)


@pytest.mark.parametrize("kernel_uni_name", kernel_uni_list)
def test_kernel_uni_raises(kernel_uni_name, request):
    k = request.getfixturevalue(kernel_uni_name)

    x = np.array([1.0, 1.0, 1.0])
    wrong_x = np.array([[1.0], [1.0]])

    # x1 has wrong shape
    with pytest.raises(ValueError):
        k.evaluate(wrong_x, x)

    # x2 has wrong shape

    with pytest.raises(ValueError):
        k.evaluate(x, wrong_x)
