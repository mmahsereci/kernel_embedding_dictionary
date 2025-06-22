# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.measures import GaussianMeasure, GaussianMeasureUni

# test that are measure specific go into this files
# - default param values
# - param values
# - raises
#
# common tests for all measures go into the common test file.
# - measure name
# - sample shapes


# tests for GaussianMeasureUni start here
def test_gaussian_measure_uni_values():

    mean = 1.4
    variance = 0.2
    m = GaussianMeasureUni(mean, variance)

    assert m.mean == mean
    assert m.variance == variance


def test_gaussian_measure_uni_param_dict():

    mean = 1.4
    variance = 0.2
    m = GaussianMeasureUni(mean, variance)
    p = m.param_dict

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"mean", "variance"}
    assert p["mean"] == mean
    assert p["variance"] == variance


def test_gaussian_measure_uni_sample_shapes():

    mean = 1.4
    variance = 0.2
    m = GaussianMeasureUni(mean, variance)
    num_points = 5
    res = m.sample(num_points)
    assert res.shape == (num_points,)


def test_gaussian_measure_uni_raises():

    mean = 0.0

    # zero variance
    wrong_variance = 0.0
    with pytest.raises(ValueError):
        GaussianMeasureUni(mean, wrong_variance)

    # negative variance
    wrong_variance = -1.0
    with pytest.raises(ValueError):
        GaussianMeasureUni(mean, wrong_variance)


# tests for GaussianMeasure start here
def test_gaussian_measure_defaults():

    # nothing given
    m = GaussianMeasure()
    assert m.ndim == 1
    assert m.means == [0.0]
    assert m.variances == [1.0]
    assert len(m._measures) == 1

    # only ndim given
    c = {"ndim": 2}
    m = GaussianMeasure(c)
    assert m.ndim == 2
    assert m.means == [0.0, 0.0]
    assert m.variances == [1.0, 1.0]
    assert len(m._measures) == 2

    # only means given
    c = {"means": [0.0, 1.0]}
    m = GaussianMeasure(c)
    assert m.ndim == 2
    assert m.means == [0.0, 1.0]
    assert m.variances == [1.0, 1.0]
    assert len(m._measures) == 2

    # only variances given
    c = {"variances": [1.0, 2.0]}
    m = GaussianMeasure(c)
    assert m.ndim == 2
    assert m.means == [0.0, 0.0]
    assert m.variances == [1.0, 2.0]
    assert len(m._measures) == 2

    # means and variances given
    c = {"means": [0.0, 1.0], "variances": [1.0, 2.0]}
    m = GaussianMeasure(c)
    assert m.ndim == 2
    assert m.means == [0.0, 1.0]
    assert m.variances == [1.0, 2.0]
    assert len(m._measures) == 2


def test_gaussian_measure_values():

    # all values given, no defaults
    c = {"ndim": 2, "means": [0.0, 1.0], "variances": [1.0, 2.0]}
    m = GaussianMeasure(c)
    assert m.ndim == 2
    assert m.means == [0.0, 1.0]
    assert m.variances == [1.0, 2.0]
    assert len(m._measures) == 2


def test_gaussian_measure_raises():

    # variances and means do not match
    wrong_c = {"means": [0.0, 1.0, 1.0], "variances": [1.0, 2.0]}
    with pytest.raises(ValueError):
        GaussianMeasure(wrong_c)

    # ndim and means do not match
    wrong_c = {"ndim": 2, "means": [0.0, 1.0, 1.0]}
    with pytest.raises(ValueError):
        GaussianMeasure(wrong_c)

    # ndim and means do not match
    wrong_c = {"ndim": 2, "variances": [1.0, 2.0, 2.0]}
    with pytest.raises(ValueError):
        GaussianMeasure(wrong_c)
