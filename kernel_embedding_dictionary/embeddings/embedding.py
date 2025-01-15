# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


class KernelEmbedding:

    # TODO: add types
    def __init__(self, kernel, measure):
        self._kernel = kernel
        self._measure = measure

    # TODO: description including kernel and measure params
    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self._kernel.__repr__}_{self._measure.__repr__}"
