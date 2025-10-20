
<p align="center">
  <img src="https://github.com/taleblou/momijo/blob/main/docs/momijo.svg" alt="Momijo logo" width="140"/>
</p>

# Momijo Tensor

> **Status: Work in Progress (WIP)** — This module is actively evolving. APIs may expand and docs/examples are being completed. **Contributions are welcome!** See the *Contributing* section below.

*A fast, NumPy-like tensor library for the **Mojo** programming language.*

> **TL;DR**: `momijo.tensor` brings a familiar, NumPy/PyTorch-style tensor API to Mojo with strong performance, strict type safety, and clean package structure. It’s designed to be a practical building block for scientific computing, ML tooling, and graphics in pure Mojo.

---

## Features

- **Core Tensor type**: `Tensor[T]` with owned storage, row‑major strides, and views.
- **DTypes**: `Int`, `Int32`, `Float32`, `Float64`, `Bool`, with explicit **promotion rules** (`Int → Float32 → Float64`).
- **Creation utilities**: `arange_*`, `zeros_like`, `full_like`, `from_list_*` (1D–3D), random tensors (`randn_*`, `rand_*`) with **seeded determinism**.
- **Indexing**: scalar/tuple indexing, slices, boolean masks, and advanced/fancy indexing (gather/scatter/take/put, boolean_select).
- **Mask ops**: `masked_select`, `masked_fill`, boolean logic (`not_bitwise`, comparisons).
- **Math & reductions**: elementwise ops, `sum/mean/min/max`, `clip` and the exhaustive `minimum_scalar` / `maximum_scalar` combinations across dtypes.
- **Shape ops**: `reshape`, `flatten(start_dim, end_dim)`, `transpose`, `squeeze/unsqueeze`, concatenate `cat`, `stack`, and splitting (`chunk`, `split_sizes`, `unbind`).
- **Linear algebra**: `@` (mv/mm), `solve`, and `inv` for small dense systems.
- **Views vs copies**: safe in‑place APIs with notes on aliasing/grad‑safety.
- **Examples**: runnable demos for all of the above.

---

## Project Layout

```
src/
  momijo/
    tensor/           # The library (Tensor[T], ops, helpers, dtypes)
examples/
  tensor/             # Runnable demos covering features end-to-end
```

The package follows strict, documented coding standards for Mojo (imports, constructors, traits, mutability, row‑major stride invariants, etc.). Headers are MIT and comments are English‑only.

---

## Install & Build

### Requirements
- **Mojo** (recent nightly recommended)
- Linux/macOS

### Clone
```bash
git clone https://github.com/taleblou/momijo
cd momijo
```

### Run examples
Each example is a single Mojo file with a `main()` entry point. Use the project include path:

```bash
mojo run -I src examples/tensor/demo_indexing.mojo
# For local side projects you may also include:
#   -I /home/morteza/life/src
```

> Tip: Tests and examples are separate; examples are intentionally small, readable, and print only scalars/strings or types that implement `__str__`.

---

## Quickstart

> The library is under active development. Please pin to a known Mojo toolchain if examples fail due to Mojo breaking changes.

```mojo
from momijo.tensor import tensor

fn main():
    # Create and reshape
    var x = tensor.arange_f64(0, 6, 1).reshape([2, 3])

    # Basic/boolean indexing
    var slice = x.slice(1, 0, 2)        # x[:, 0:2]
    var mask  = x.gt_scalar(0.0)

    # Mask utilities
    var sel = x.masked_select(mask)     # flatten selection
    var y = x.copy()
    y.masked_fill(mask.not_bitwise(), 0)

    # Join/split
    var z = tensor.arange_f64(100, 106, 1).reshape([2, 3])
    var cat0 = tensor.cat([x.copy(), z.copy()], 0)
    var st1  = tensor.stack([x.copy(), z.copy()], 0)  # shape [2,2,3]

    # Linear algebra
    var b = tensor.randn_f64([3], 123)
    var Ab = x.reshape([3,2]).transpose([1,0]) @ b     # mv

    print("x:\n" + x.__str__())
    print("sel (>0):\n" + sel.__str__())
    print("cat dim=0:\n" + cat0.__str__())
    print("A@b (mv): " + Ab.__str__())
```

---

## API Overview

> Namespaces below are exposed via `from momijo.tensor import tensor`.

### Creation
- `arange_*` (`arange_int`, `arange_f32`, `arange_f64`)
- `zeros_like`, `full_like`
- `from_list_*` (1D and shape‑aware constructors)
- Random: `randn_*`, `rand_*` with integer/float variants and **seed** argument

### Indexing & Advanced Ops
- Square‑bracket indexing for scalars and slices (e.g., `x[0, 1]`, `x[:, 1:3]`)
- Boolean masks: `eq_scalar`, `gt_scalar`, `lt_scalar`, etc., logical ops
- Advanced: `gather`, `scatter`, `scatter_add`, `take`, `put`, `boolean_select`

### Shape & Join/Split
- `reshape`, `flatten(start_dim, end_dim)`, `transpose`
- `squeeze`, `unsqueeze`
- `cat(tensors, dim)`, `stack(tensors, dim)`
- `chunk(t, chunks, dim)`, `split_sizes(t, sizes, dim)`, `unbind(t, dim)`

### Math & Reductions
- Elementwise: `add/sub/mul/div`, `pow`, comparisons
- Reductions: `sum`, `mean`, `min`, `max`
- Clamp/clip family:
  - `clamp` (in‑place) and `clamped` (out‑of‑place)
  - Exhaustive combinations of `minimum_scalar` / `maximum_scalar` across `Int`, `Float32`, `Float64`

### Linear Algebra
- `@` operator (mv/mm)
- `solve(A, b)` and `inv(A)` for small dense matrices (sanity‑checked in examples)

### Randomness & Seeds
- Random APIs accept a `seed: Int` for **deterministic** results
- Example: drawing two tensors with the same seed yields identical values

---

## Capabilities by Example File (living index)

> This list is maintained directly from the files under `examples/tensor/`. As new demos land, this section will grow. PRs to keep it in sync are appreciated.

### `demo_indexing.mojo`
- Basic indexing (`x[i, j]`), slicing (`x[:, a:b]`)
- Boolean masks (`eq/gt/lt` + `not_bitwise`)
- Advanced indexing: `gather`, `scatter`, `scatter_add`, `take`, `put`

### `demo_join_split.mojo`
- Join: `cat(tensors, dim)`, `stack(tensors, dim)`
- Split: `chunk`, `split_sizes`, `unbind`

### `demo_inplace_notes.mojo`
- Views vs copies; safe in‑place operations
- Aliasing caveats and out‑of‑place equivalents

### `demo_masks_fill_clamp.mojo`
- `masked_select`, `masked_fill`
- `clamp` (in‑place) and `clamped` (out‑of‑place) across dtypes

### `demo_masked_reductions.mojo`
- Per‑class reductions using boolean masks (e.g., masked `mean`)

### `demo_linear_algebra.mojo` / `demo_solve.mojo`
- Matrix–vector / matrix–matrix products via `@`
- Linear solves `solve(A, b)` and matrix inverse `inv(A)`

### `demo_setitem_numpy_parity.mojo`
- NumPy‑style `__setitem__` parity across 1D…5D selectors
- String‑spec and native selector coverage

### `demo_shape_ops.mojo`
- `reshape`, `flatten(start_dim, end_dim)`, `transpose`
- `squeeze`, `unsqueeze`

### `demo_seeds_and_determinism.mojo`
- RNG seeding; reproducible `rand` / `randn` draws

> If a filename differs slightly in your tree, the capability remains the same; the demos exercise these exact operations.

---

## Design Notes

- **Row‑major, safe strides** with zero‑copy slices when possible.
- **Views vs copies**: in‑place ops are available when safe; examples show aliasing caveats.
- **Promotion rules** are explicit: `Int → Float32 → Float64` for mixed‑type math.
- **Traits & Generics**: `Tensor[T]` bounds (`ImplicitlyCopyable & Copyable & Movable`) are used consistently.
- **Printing**: only scalars/strings or types with `__str__` to keep examples robust.

---

## Roadmap

- Expanded linear algebra (decompositions, batched ops)
- Broadcasting across all elementwise ops
- Performance tuning for non‑contiguous views
- Extended dtype surface (UInts, complex) when Mojo permits

---

## Contributing

We’re building `momijo.tensor` in the open and **actively welcoming contributors** of all experience levels.

### How to contribute
1. **Fork** the repo and create a feature branch (`feat/<topic>` or `fix/<topic>`).
2. **Follow the coding standards** (Mojo‑specific) used in this project:
   - *Var‑only; no globals;* constructors as `fn __init__(out self, ...)`
   - No wildcard imports; import exact submodules
   - Traits/generics bounds respected (`ImplicitlyCopyable & Copyable & Movable`)
   - Row‑major strides, zero‑copy slices where possible
   - English‑only comments; MIT short header on every source file
3. **Add a runnable example** under `examples/tensor/` when adding a new user‑facing feature.
4. **Run lint & smoke tests** (see CI config or `tools/mojo_lint.py`; smoke: `mojo run -I src tests/smoke.mojo`).
5. Open a **PR** with a concise description, perf notes (if any), and checkboxes for:
   - [ ] API documented in README / example updated
   - [ ] No wildcard imports / correct module paths
   - [ ] Constructors follow `out self` rule
   - [ ] Views/copies semantics tested (if touching indexing/shape)
   - [ ] Printability (`__str__`) maintained for demo output

### Code Review Guidelines
- Prefer **clear, small diffs** and **deduplicated** implementations (especially for dtype combos).
- Keep **promotion rules** explicit (`Int → Float32 → Float64`).
- Avoid pointer/unsafe patterns unless necessary; document invariants if used.
- Ensure **contiguous and non‑contiguous** paths are covered where performance matters.

> **WIP**: Docs and examples are being expanded. If you see gaps, open an issue or PR—your feedback drives the roadmap.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

```
Copyright (c) 2025 Morteza Taleblou & Mitra Daneshmand
Website: https://taleblou.ir/
Repository: https://github.com/taleblou/momijo
```

---

## Acknowledgements

- Mojo language and community
- Inspiration from NumPy/PyTorch ergonomics, adapted to Mojo’s type system

---

## Citation

If you use **Momijo Tensor** in academic work, please cite the repository:

```text
@software{momijo_tensor,
  author  = {Morteza Taleblou and Mitra Daneshmand},
  title   = {Momijo Tensor: A NumPy-like Tensor Library for Mojo},
  year    = {2025},
  url     = {https://github.com/taleblou/momijo}
}
```
