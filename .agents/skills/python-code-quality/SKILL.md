---
name: python-code-quality
description: Professional Python engineering standard and code-quality gatekeeper. Enforces static typing with mypy, lightning-fast linting and formatting with ruff, explicit error handling, immutability invariants, and strict test coverage with pytest.
---

# Python Code Quality Skill

This skill enforces enterprise-grade Python software engineering standards, clean architecture principles, and robust typing across all machine learning, control systems, and quantitative pipelines.

---

## 1. Static Type Checking & Modern Type Annotations (Mypy)

Every function, class, and method must have explicit type annotations for parameters and return values.

### Strict Typing Rules:
- **Modern Union Syntax**: Use `T | None` and `A | B` instead of `Optional[T]` and `Union[A, B]` (Python 3.10+).
- **NumPy Typing**: Use `numpy.typing.NDArray[np.float64]` or `NDArray[np.float64 | np.complex128]` rather than bare `np.ndarray`.
- **Avoid Bare `Any`**: Replace `Any` with concrete types, protocols, generics (`TypeVar`), or narrow unions whenever type bounds are knowable.
- **Explicit Factory Return Types**: Method chaining and immutable copy constructors must return `Self` (from `typing import Self`) or the explicit class name.

```python
# GOOD: Fully annotated, explicit NumPy typing and modern union
from typing import Self
import numpy as np
from numpy.typing import NDArray

def compute_moving_zscore(
    series: NDArray[np.float64], 
    window: int, 
    min_periods: int | None = None
) -> NDArray[np.float64]:
    ...
```

---

## 2. Formatting, Import Organization & Linting (Ruff)

Adhere to modern PEP 8 standards with automated import organization:

### Import Hierarchy (Strict PEP 8 grouping separated by single blank lines):
1. **Standard Library**: `os`, `sys`, `math`, `typing`, `dataclasses`, `warnings`.
2. **Third-Party Scientific & Quantitative**: `numpy`, `scipy`, `pandas`, `ccxt`, `torch`, `statsmodels`.
3. **Local Project Modules**: Relative or absolute local package imports (e.g., `from core.validation import ...`).

### Linting Guardrails:
- **No Wildcard Imports**: Never use `from module import *` in library or production modules. Always import explicit symbols or the module namespace.
- **Line Length**: Target 88 to 100 characters per line.
- **Dead Code & Unused Variables**: Prefix unused variables with an underscore (e.g., `_idx`, `_`). Remove unused imports.
- **Docstrings**: Use Google-style docstrings for public APIs specifying `Args:`, `Returns:`, and `Raises:`.

---

## 3. Exception Handling & Architectural Error Boundaries

- **No Bare `except:`**: Never catch bare `except:` or catch `Exception` without re-raising or logging with root cause.
- **Domain Exceptions**: Use specific domain exception hierarchies (e.g., `SystemIntegrityError`, `RiskLimitExceededError`).
- **Exception Chaining**: Always preserve original tracebacks when wrapping lower-level exceptions:
  ```python
  # GOOD: Preserves original LinAlgError context
  try:
      inv_matrix = np.linalg.inv(matrix)
  except np.linalg.LinAlgError as err:
      raise SystemIntegrityError("Matrix inversion failed: singular matrix.") from err
  ```
- **Distinguish Numerical Diagnostics from Hard Errors**:
  - Invalid dimensions, non-numeric inputs, or NaN/Inf $\rightarrow$ Raise custom `Exception`.
  - Ill-conditioned but mathematically valid physical states (e.g. pure integrators, high condition numbers) $\rightarrow$ Emit `warnings.warn(..., CategoryWarning)`.

---

## 4. Immutability & Mathematical Integrity

- **Defensive Copying**: Properties exposing internal NumPy arrays must return copies (`return self._data.copy()`) to prevent external callers from silently mutating internal state.
- **No Silent Downcasting**: Never cast complex numbers to float without explicit validation or explicit warning/user intent.
- **Centralized Tolerances**: Never hardcode arbitrary magic numbers (e.g. `1e-7`, `1e-5`, `1e-12`) scattered across files. Import them from a centralized policy module (e.g., `core.validation` or `config.settings`).

---

## 5. Verification Protocol (Pytest & Linters)

Before committing or concluding a task, run the automated verification gate:

```powershell
# 1. Automated formatting and lint check
ruff check .
ruff format --check .

# 2. Strict static type analysis
mypy . --ignore-missing-imports

# 3. Comprehensive test suite execution with warnings inspection
pytest -v
```

### Test Authoring Invariants:
- Floating-point assertions must use `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)` instead of equality (`==`).
- Every new function must have at least one nominal benchmark test and one boundary/exception test (`pytest.raises`).
