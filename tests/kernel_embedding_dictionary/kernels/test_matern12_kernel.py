# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.kernels import Matern12Kernel, Matern12KernelUni


# tests for Matern12KernelUni start here
def test_matern12_kernel_uni_values():

    ell = 1.5
    k = Matern12KernelUni(ell)
    assert k.ell == ell
    assert k.nu == 0.5


def test_matern12_kernel_uni_param_dict():

    ell = 1.5
    k = Matern12KernelUni(ell)
    p = k.param_dict

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"ell", "nu"}
    assert p["ell"] == ell
    assert p["nu"] == 0.5


def test_matern12_kernel_uni_raises():

    # negative lengthscale
    wrong_ell = -1.0
    with pytest.raises(ValueError):
        Matern12KernelUni(wrong_ell)

    # zero lengthscale
    wrong_ell = 0.0
    with pytest.raises(ValueError):
        Matern12KernelUni(wrong_ell)


# tests for Matern12Kernel start here
def test_matern12_kernel_defaults():

    # nothing given
    k = Matern12Kernel()
    assert k.ndim == 1
    assert k.ell == [1.0]
    assert len(k._kernels) == 1
    assert k.nu == 0.5

    # only ndim given
    c = {"ndim": 2}
    k = Matern12Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 1.0]
    assert len(k._kernels) == 2
    assert k.nu == 0.5

    # only ell given
    c = {"lengthscales": [1.0, 2.0]}
    k = Matern12Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 0.5


def test_matern12_kernel_values():

    # all values given, no defaults
    c = {"ndim": 2, "lengthscales": [1.0, 2.0]}
    k = Matern12Kernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 0.5


def test_matern12_kernel_raises():

    # ndim and lengthscales do not match
    wrong_c = {"ndim": 1, "lengthscales": [1.0, 1.0]}
    with pytest.raises(ValueError):
        Matern12Kernel(wrong_c)
