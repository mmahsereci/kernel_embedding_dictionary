# Code Review: kernel_embedding_dictionary

Reviewed by Claude Sonnet 4.6 | 2026-05-27 | Scope: all source, tests, CI, tooling

---

## Summary

Overall this is a clean, well-structured library. The core mathematical idea (product decomposition → 1D
formulas) maps neatly onto the code, the public API is ergonomic, and the test strategy is sound. Remaining
actionable issues are: two places where the same enumeration of kernel-measure pairs is duplicated, opaque
error messages, missing numerical accuracy tests for the generic Matern embedding, and minor cleanliness
items (print statements in tests, access to private attributes).

---

## Design issues

### 1. Duplicated kernel-measure registry (main structural issue)

The same enumeration of valid `"kernel-measure"` pairs appears in two places:

- `kernel_embedding_dictionary/_get_embedding.py` → `available_embeddings_dict` (maps string to kernel/measure classes)
- `kernel_embedding_dictionary/embeddings/embedding.py` → `mean_func_1d_dict` in `_get_1d_funcs()` (maps string to mean function)

Both must be updated in sync when adding a new embedding. Both dicts are rebuilt on every call (they're
local to the method/function). The AGENTS.md PR checklist already documents this, but it's a genuine
inconsistency risk.

A cleaner approach: have one module-level dict (or a dataclass) that maps each string key to a tuple of
`(KernelClass, MeasureClass, mean_func_1d)`. Then `get_embedding` and `KernelEmbedding` both look up
from that single source of truth. This is a refactor worth doing if you add more embeddings.

---

### 5. `KernelEmbedding` re-validates at construction, duplicating `get_embedding`'s check

`KernelEmbedding._get_1d_funcs` raises the same "unknown" error if the kernel-measure combination isn't
in the internal dict. When called through `get_embedding` this can never happen (the factory already
checked). When called directly with a valid-but-unregistered pair, the error message is confusing
because it doesn't match the factory error. This is a minor issue, but it's worth noting that the
two-dict architecture is what forces this double-check.

---

## Test issues

### 7. `compute_mean_intervals.py` doesn't include `matern-lebesgue`

The `embedding_to_config_list_dict` in the script is missing the `"matern-lebesgue"` key, so you can't
run the script to regenerate bounds for that combo without manually editing it. Should be added, even if
the bounds aren't yet used in tests.

---

### 10. `compute_mean_intervals.py` uses bare local imports

```python
from test_mean_values_expquad import ...
```

This works only when the script is run from its own directory (`tests/kernel_embedding_dictionary/embeddings/`).
It breaks if run from anywhere else. Not a correctness issue since it's a dev utility, but adding a
`sys.path` manipulation or converting it to use absolute imports from `kernel_embedding_dictionary` would
make it more robust.

---

## Tooling and automation

### CI

**What's already there — good:**
- `tests.yml` runs pytest on Python 3.10, 3.11, 3.12, 3.13 on push/PR to main.
- `formatting.yml` checks isort + black on push/PR to main.
- `codecov.yml` uploads coverage to Codecov.

**Gaps:**

**A. CI only triggers on `main`.** Both workflows only trigger on push/PR to `main`. This means
feature branches get no CI feedback until you open a PR. Adding `branches: ["*"]` or removing the
branch filter entirely would run CI on every push. Very low effort change.

---

### Versioning

Version is hardcoded at `0.0.1` in `pyproject.toml`. For a personal library with no PyPI publish
ambitions this is completely fine. If you ever want to track changes more formally, the lightest
approach is:

1. Keep a `CHANGELOG.md` with a one-line entry per meaningful change.
2. Bump the version manually in `pyproject.toml` when you tag a release.

There's no need for `bumpversion`, `semantic-release`, or any tooling here. Just do it manually when
it matters.

---

## What's done well

- **The 1D decomposition is the right abstraction.** Everything reduces to scalar operations, which
  keeps the mathematical formulas readable and the code easy to verify. The `scaled_diff` helper is a
  good example — small, named, reusable.

- **The public API is clean.** `get_embedding(kernel_name, measure_name)` with optional config dicts is
  easy to discover and use. Sensible defaults (1D standard kernel/measure) make the zero-config case
  trivial. This is the right shape for a dictionary-style library.

- **CI is substantially in place.** Three workflows covering tests (multi-Python matrix), formatting,
  and coverage is more than most small research libraries have.

- **Test strategy is sound.** Separating generic interface tests (`test_kernels.py`, `test_measures.py`)
  from kernel-specific tests, and using pre-computed MC intervals for numerical accuracy, is a good
  approach. The MC interval idea specifically is clever: it makes the tests fast and deterministic while
  still validating the analytical formula against simulation.

- **The PR checklist in AGENTS.md is accurate and complete.** It correctly identifies every file that
  needs touching for each type of change. This kind of checklist is very useful for avoiding the "forgot
  to register it" class of bugs.

- **The `matern` generic kernel test is a useful consistency check.** Testing that `matern(nu=0.5)`
  produces the same result as `matern12` is a smart cross-validation that doesn't require new MC
  estimates.

- **License headers on all files.** Consistent and correctly formatted.

- **The README is genuinely useful.** It covers installation, usage (with and without config), the full
  support matrix, per-kernel config documentation, and contributing instructions. For a research library,
  this is the right level of detail.

---

## Open items

| # | What | Effort | Impact |
|---|------|--------|--------|
| 1 | Consolidate the two registration dicts | Refactor | Structural |
