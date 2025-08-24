# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from kernel_embedding_dictionary.kernels import (
    Matern12Kernel,
    Matern12KernelUni,
    Matern32Kernel,
    Matern32KernelUni,
    Matern52Kernel,
    Matern52KernelUni,
    Matern72Kernel,
    Matern72KernelUni,
    MaternKernel,
    MaternKernelUni,
)


# tests for MaternKernelUni start here
def test_matern_kernel_uni_values():

    ell = 1.5
    nu = 7.5
    k = MaternKernelUni(nu=nu, ell=ell)
    assert k.ell == ell
    assert k.nu == nu


def test_matern_kernel_uni_param_dict():

    ell = 1.2
    nu = 0.5
    k = MaternKernelUni(nu=nu, ell=ell)
    p = k.param_dict

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"ell", "nu"}
    assert p["ell"] == ell
    assert p["nu"] == nu


def test_matern_kernel_uni_raises():

    # negative lengthscale
    wrong_ell = -1.0
    with pytest.raises(ValueError):
        MaternKernelUni(nu=0.5, ell=wrong_ell)

    # zero lengthscale
    wrong_ell = 0.0
    with pytest.raises(ValueError):
        MaternKernelUni(nu=0.5, ell=wrong_ell)

    # negative nu
    wrong_nu = -0.5
    with pytest.raises(ValueError):
        MaternKernelUni(nu=wrong_nu, ell=1.0)

    # non-half-integer nu
    wrong_nu = 2.55
    with pytest.raises(ValueError):
        MaternKernelUni(nu=wrong_nu, ell=1.0)


def test_matern_kernel_uni_evaluations():

    ell = 1.8
    x = np.array([0, -1.0, 0.123, 3.144365, 2, 1.0])

    # Matern 1/2
    k1 = Matern12KernelUni(ell=ell)
    k2 = MaternKernelUni(nu=0.5, ell=ell)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # Matern 3/2
    k1 = Matern32KernelUni(ell=ell)
    k2 = MaternKernelUni(nu=1.5, ell=ell)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # Matern 5/2
    k1 = Matern52KernelUni(ell=ell)
    k2 = MaternKernelUni(nu=2.5, ell=ell)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # Matern 7/2
    k1 = Matern72KernelUni(ell=ell)
    k2 = MaternKernelUni(nu=3.5, ell=ell)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all


# tests for MaternKernel start here
def test_matern_kernel_defaults():

    # nothing given
    k = MaternKernel()
    assert k.ndim == 1
    assert k.ell == [1.0]
    assert len(k._kernels) == 1
    assert k.nu == 2.5

    # only ndim given
    c = {"ndim": 3}
    k = MaternKernel(c)
    assert k.ndim == 3
    assert k.ell == [1.0, 1.0, 1.0]
    assert len(k._kernels) == 3
    assert k.nu == 2.5

    # only nu given
    c = {"nu": 1.5}
    k = MaternKernel(c)
    assert k.ndim == 1
    assert k.ell == [1.0]
    assert len(k._kernels) == 1
    assert k.nu == 1.5

    # only nu and ndim given
    c = {"nu": 2.5, "ndim": 2}
    k = MaternKernel(c)
    assert k.ndim == 2
    assert k.ell == [1.0, 1.0]
    assert len(k._kernels) == 2
    assert k.nu == 2.5

    # only ell and ndim given
    c = {"lengthscales": [1.3, 2.0, 0.5, 0.5], "ndim": 4}
    k = MaternKernel(c)
    assert k.ndim == 4
    assert k.ell == [1.3, 2.0, 0.5, 0.5]
    assert len(k._kernels) == 4
    assert k.nu == 2.5

    # only ell given
    c = {"lengthscales": [1.0, 2.0]}
    k = MaternKernel(config=c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 2.5


def test_matern_kernel_values():

    # all values given, no defaults
    c = {"nu": 3.5, "ndim": 2, "lengthscales": [1.0, 2.0]}
    k = MaternKernel(config=c)
    assert k.ndim == 2
    assert k.ell == [1.0, 2.0]
    assert len(k._kernels) == 2
    assert k.nu == 3.5


def test_matern_kernel_raises():

    # ndim and lengthscales do not match
    wrong_c = {"nu": 0.5, "ndim": 1, "lengthscales": [1.0, 1.0]}
    with pytest.raises(ValueError):
        MaternKernel(wrong_c)


def test_matern_kernel_evaluations():

    x = np.array([[0, 0], [-1.5, -1.0], [1.0, 0.123], [2, 3.144365], [2, 3], [1.0, 1]])

    # Matern 1/2
    c1 = {"ndim": 2, "lengthscales": [1.0, 0.5]}
    c2 = {"nu": 0.5, "ndim": 2, "lengthscales": [1.0, 0.5]}
    k1 = Matern12Kernel(c1)
    k2 = MaternKernel(c2)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # # Matern 3/2
    c1 = {"ndim": 2, "lengthscales": [1.0, 0.5]}
    c2 = {"nu": 1.5, "ndim": 2, "lengthscales": [1.0, 0.5]}
    k1 = Matern32Kernel(c1)
    k2 = MaternKernel(c2)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # # Matern 5/2
    c1 = {"ndim": 2, "lengthscales": [1.0, 0.5]}
    c2 = {"nu": 2.5, "ndim": 2, "lengthscales": [1.0, 0.5]}
    k1 = Matern52Kernel(c1)
    k2 = MaternKernel(c2)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all

    # # Matern 7/2
    c1 = {"ndim": 2, "lengthscales": [1.0, 0.5]}
    c2 = {"nu": 3.5, "ndim": 2, "lengthscales": [1.0, 0.5]}
    k1 = Matern72Kernel(c1)
    k2 = MaternKernel(c2)
    assert (k1.evaluate(x, x) == k2.evaluate(x, x)).all
