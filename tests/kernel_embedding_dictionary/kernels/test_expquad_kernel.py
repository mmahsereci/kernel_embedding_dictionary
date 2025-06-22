# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.kernels import ExpQuadKernel, ExpQuadKernelUni


# tests for ExpQuadKernelUni start here
def test_expquad_kernel_uni_values():

    ell = 1.5
    k = ExpQuadKernelUni(ell)
    assert k.ell == ell


def test_expquad_kernel_uni_param_dict():

    ell = 1.5
    k = ExpQuadKernelUni(ell)
    p = k.param_dict

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"ell"}
    assert p["ell"] == ell


def test_expquad_kernel_uni_raises():

    # negative lengthscale
    wrong_ell = -1.0
    with pytest.raises(ValueError):
        ExpQuadKernelUni(wrong_ell)

    # zero lengthscale
    wrong_ell = 0.0
    with pytest.raises(ValueError):
        ExpQuadKernelUni(wrong_ell)


# tests for ExpQuadKernel start here
def test_expquad_kernel_defaults():

    # nothing given
    k = ExpQuadKernel()
    assert k.ndim == 1
    assert k.ell == [1.0]
    assert len(k._kernels) == 1

    # only ndim given
    c = {"ndim": 2}
    k = ExpQuadKernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 1.0]
    assert len(k._kernels) == 2

    # only ell given
    c = {"lengthscales": [1.0, 2.0]}
    k = ExpQuadKernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2


def test_expquad_kernel_values():

    # all values given, no defaults
    c = {"ndim": 2, "lengthscales": [1.0, 2.0]}
    k = ExpQuadKernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2


def test_expquad_kernel_raises():

    # ndim and lengthscales do not match
    wrong_c = {"ndim": 1, "lengthscales": [1.0, 1.0]}
    with pytest.raises(ValueError):
        ExpQuadKernel(wrong_c)
