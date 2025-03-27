# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.kernels import Matern32Kernel, Matern32KernelUni


# tests for Matern32KernelUni start here
def test_matern32_kernel_uni_values():

    ell = 1.5
    k = Matern32KernelUni(ell)
    assert k.ell == ell
    assert k.nu == 1.5


def test_matern32_kernel_uni_param_dict():

    ell = 1.5
    k = Matern32KernelUni(ell)
    p = k.get_param_dict()

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"ell", "nu"}
    assert p["ell"] == ell
    assert p["nu"] == 1.5


def test_matern32_kernel_uni_raises():

    # negative lengthscale
    wrong_ell = -1.0
    with pytest.raises(ValueError):
        Matern32KernelUni(wrong_ell)

    # zero lengthscale
    wrong_ell = 0.0
    with pytest.raises(ValueError):
        Matern32KernelUni(wrong_ell)


# tests for Matern32Kernel start here
def test_matern32_kernel_defaults():

    # nothing given
    k = Matern32Kernel()
    assert k.ndim == 1
    assert k.ell == [1.0]
    assert len(k._kernels) == 1
    assert k.nu == 1.5

    # only ndim given
    c = {"ndim": 2}
    k = Matern32Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 1.0]
    assert len(k._kernels) == 2
    assert k.nu == 1.5

    # only ell given
    c = {"lengthscales": [1.0, 2.0]}
    k = Matern32Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 1.5


def test_matern32_kernel_values():

    # all values given, no defaults
    c = {"ndim": 2, "lengthscales": [1.0, 2.0]}
    k = Matern32Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 1.5


def test_matern32_kernel_raises():

    # ndim and lengthscales do not match
    wrong_c = {"ndim": 1, "lengthscales": [1.0, 1.0]}
    with pytest.raises(ValueError):
        Matern32Kernel(wrong_c)
