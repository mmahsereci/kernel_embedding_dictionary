# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest


REL_TOL = 1e-5
ABS_TOL = 1e-4

from kernel_embedding_dictionary.measures import LebesgueMeasure, LebesgueMeasureUni


@pytest.fixture
def lebesgue_measure_uni():
    lb = 0.0
    ub = 1.0
    normalize = False
    return LebesgueMeasureUni(lb, ub, normalize)


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


def test_lebesgue_measure_uni_raises():

    # upper bound smaller than lower bound
    lb = 2.0
    ub = 1.0
    with pytest.raises(ValueError):
        LebesgueMeasureUni(lb, ub, normalize=True)


# tests for LebesgueMeasure start here
def test_lebesgue_measure_defaults():
    pass

#    # noting given
#    m = LebesgueMeasure()
#    assert m.
#

#    config = {
#        "ndim": 2,
#        "bounds": [(0, 1)],
#        "normalize": True
#    }

def test_lebesgue_measure_values():
    pass

