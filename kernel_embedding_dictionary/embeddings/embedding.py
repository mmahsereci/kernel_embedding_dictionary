# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


from ..kernels import Kernel
from ..measures import Measure


class KernelEmbedding:
    def __init__(self, kernel: Kernel, measure: Measure):
        self._kernel = kernel
        self._measure = measure

    def __str__(self) -> str:

        return f"Kernel embedding for\n\n{self._kernel.__str__()}\n\nand\n\n{self._measure.__str__()}"

    def __repr__(self) -> str:
        return f"Kernel embedding for {self._kernel.__repr__()} and {self._measure.__repr__()}."
