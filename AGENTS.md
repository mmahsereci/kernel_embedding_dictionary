# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e .[dev]

# Run all tests
pytest

# Run a single test file
pytest tests/kernel_embedding_dictionary/kernels/test_expquad_kernel.py

# Run a single test
pytest tests/test_get_embedding.py::test_get_embedding_raises

# Format code (run isort before black)
isort .
black .

# Check formatting without modifying files
isort --check .
black --check .
```

## Architecture

The library computes closed-form kernel mean embeddings $\int k(x, \cdot)\, dp$ for a fixed set of kernel–measure pairs. All embeddings decompose as products over dimensions, so the implementation is entirely 1D under the hood.

**Three core modules**, each following the same pattern:

- `kernels/` — `UnivariateKernel` (abstract, evaluates $k(x_1, x_2)$) → `ProductKernel` (holds a list of univariate kernels, evaluates the product). Raw 1D kernel functions live in `kernels/kernel_funcs_1d.py`.
- `measures/` — `UnivariateMeasure` (abstract, holds params + can sample) → `ProductMeasure`. Unlike kernels, each measure class contains its own logic rather than delegating to a shared functions file.
- `embeddings/` — `KernelEmbedding` wraps a `ProductKernel` + `ProductMeasure`. Its `mean(x)` method loops over dimensions and multiplies 1D mean values. All closed-form 1D mean embedding formulas live in `embeddings/mean_funcs_1d.py`.

**Public API** is a single factory function: `get_embedding(kernel_name, measure_name, kernel_config, measure_config)` in `_get_embedding.py`. It looks up the kernel–measure pair by the string key `"<kernel>-<measure>"` and returns a `KernelEmbedding`.

For all changes, follow the PR Checklist below — it is the authoritative source for what files to touch.

## License Header

Every new `.py` file must start with this two-line header. Replace `<YEAR>` with the **current calendar year at the time the file is created** (e.g. a file created in 2026 gets `2026`, one created in 2027 gets `2027`):

```python
# Copyright <YEAR> The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT
```

## Tests

Test structure mirrors the source tree under `tests/kernel_embedding_dictionary/`. Embedding value tests compare analytic results against pre-computed Monte Carlo estimates. To generate those estimates for a new embedding, use `tests/kernel_embedding_dictionary/embeddings/compute_mean_intervals.py` and copy the output into the corresponding `test_mean_values_<kernel>.py` file as `mean_intervals`.

## PR Checklist

**Every PR:**
- [ ] Run `isort .` then `black .`; confirm no diffs remain
- [ ] `pytest tests/` passes locally
- [ ] Every new `.py` file has the correct license header with the current year
- [ ] PR body mentions that the PR was co-authored by an AI agent

**When adding a new kernel:**
- [ ] Univariate kernel function added to `kernels/kernel_funcs_1d.py`
- [ ] New module `kernels/<name>_kernel.py` implementing `UnivariateKernel` and `ProductKernel`
- [ ] Kernel registered in `kernels/__init__.py`
- [ ] Kernel added as fixture to `tests/kernel_embedding_dictionary/kernels/test_kernels.py` and `test_kernels_uni.py`
- [ ] Dedicated `tests/kernel_embedding_dictionary/kernels/test_<name>_kernel.py` created

**When adding a new measure:**
- [ ] New module `measures/<name>_measure.py` implementing `UnivariateMeasure` and `ProductMeasure`
- [ ] Measure registered in `measures/__init__.py`
- [ ] Measure added as fixture to `tests/kernel_embedding_dictionary/measures/test_measures.py`
- [ ] Dedicated `tests/kernel_embedding_dictionary/measures/test_<name>_measure.py` created

**When adding a new kernel–measure embedding:**
- [ ] Closed-form 1D mean function added to `embeddings/mean_funcs_1d.py`
- [ ] Combination registered in `embeddings/embedding.py` (`mean_func_1d_dict`) and `_get_embedding.py` (`available_embeddings_dict`)
- [ ] Combination added to `tests/test_get_embedding.py`
- [ ] Monte Carlo estimates generated via `tests/kernel_embedding_dictionary/embeddings/compute_mean_intervals.py` and added to `tests/kernel_embedding_dictionary/embeddings/test_mean_values_<kernel>.py`
- [ ] Support matrix table in `README.md` updated
