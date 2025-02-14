# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import pytest

from kernel_embedding_dictionary.measures import LebesgueMeasure, LebesgueMeasureUni

# test that are measure specific go into this files
# - default param values
# - param values
# - raises
#
# common tests for all measures go into the common test file.
# - measure name
# - sample shapes


# tests for LebesgueMeasureUni start here
def test_lebesgue_measure_uni_values():

    # uni
    lb = 2.0
    ub = 4.5
    density = 1 / (ub - lb)

    # uni not normalized
    m = LebesgueMeasureUni(lb, ub, normalize=False)
    assert m.lb == lb
    assert m.ub == ub
    assert not m.normalize
    assert m.density == 1.0

    # uni normalized
    m = LebesgueMeasureUni(lb, ub, normalize=True)
    assert m.lb == lb
    assert m.ub == ub
    assert m.normalize
    assert m.density == density


def test_lebesgue_measure_uni_param_dict():

    lb = 2.0
    ub = 4.5
    density = 1 / (ub - lb)
    m = LebesgueMeasureUni(lb, ub, normalize=True)
    p = m.get_param_dict()

    # this check is important due to how the kernel embedding params are assembled
    assert set(p.keys()) == {"lb", "ub", "density"}
    assert p["lb"] == lb
    assert p["ub"] == ub
    assert p["density"] == density


def test_lebesgue_measure_uni_sample_shapes():

    lb = 2.0
    ub = 4.5
    m = LebesgueMeasureUni(lb, ub, normalize=True)
    num_points = 5
    res = m.sample(num_points)
    assert res.shape == (num_points,)


def test_lebesgue_measure_uni_raises():

    # upper bound smaller than lower bound
    lb = 2.0
    ub = 1.0
    with pytest.raises(ValueError):
        LebesgueMeasureUni(lb, ub, normalize=True)


# tests for LebesgueMeasure start here
def test_lebesgue_measure_defaults():

    # nothing given
    m = LebesgueMeasure()
    assert m.ndim == 1
    assert m.lb == [0.0]
    assert m.ub == [1.0]
    assert m.bounds == [(0.0, 1.0)]
    assert not m.normalize
    assert len(m._measures) == 1

    # only ndim given
    c = {"ndim": 2}
    m = LebesgueMeasure(c)
    assert m.ndim == 2
    assert m.lb == [0.0, 0.0]
    assert m.ub == [1.0, 1.0]
    assert m.bounds == [(0.0, 1.0), (0.0, 1.0)]
    assert not m.normalize
    assert len(m._measures) == 2

    # only bounds given
    c = {"bounds": [(0.0, 1.5), (1.0, 2.0)]}
    m = LebesgueMeasure(c)
    assert m.ndim == 2
    assert m.lb == [0.0, 1.0]
    assert m.ub == [1.5, 2.0]
    assert m.bounds == [(0.0, 1.5), (1.0, 2.0)]
    assert not m.normalize
    assert len(m._measures) == 2

    # ndim and 1D bounds given
    c = {"ndim": 2, "bounds": [(1.0, 2.5)]}
    m = LebesgueMeasure(c)
    assert m.ndim == 2
    assert m.lb == [1.0, 1.0]
    assert m.ub == [2.5, 2.5]
    assert m.bounds == [(1.0, 2.5), (1.0, 2.5)]
    assert not m.normalize
    assert len(m._measures) == 2


def test_lebesgue_measure_values():

    # all values given, no defaults
    c = {"ndim": 2, "bounds": [(1.0, 2.5), (0.0, 1.0)], "normalize": True}
    m = LebesgueMeasure(c)
    assert m.ndim == 2
    assert m.lb == [1.0, 0.0]
    assert m.ub == [2.5, 1.0]
    assert m.bounds == [(1.0, 2.5), (0.0, 1.0)]
    assert m.normalize
    assert len(m._measures) == 2


def test_lebesgue_measure_raises():

    # ndim and bounds do not match
    wrong_c = {"ndim": 1, "bounds": [(0.0, 1.0), (0.0, 1.0)]}
    with pytest.raises(ValueError):
        LebesgueMeasure(wrong_c)
