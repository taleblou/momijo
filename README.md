<!-- SPDX-FileCopyrightText: 2026 Morteza Taleblou -->
<!-- SPDX-FileCopyrightText: 2026 Mitra Daneshmand -->
<!-- SPDX-License-Identifier: MIT -->

<div align="center">

<img src="assets/momijo.svg" alt="Momijo logo" width="156">

# Momijo

Two independent tensor engines written in Mojo.

</div>

| Engine | CPU APIs |
| --- | --- |
| General (`momijo.tensor`) | Owned dense row-major Float32 tensors with runtime rank. Create scalars with `TensorF32(value)` or create a tensor with `RuntimeTensorShape.try_from_dims` and `ones_f32`. Use `get`, `set`, `reshape_in_place`, `sum_all_f32`, and `mean_all_f32`. |
| Light (`momijo.tensor_light`) | Float32 tensors with compile-time rank 0–8. Use `TensorShape[rank].checked` and `zeros[rank]` or `ones[rank]` for host tensors. The CPU view APIs support slicing, transposing, permuting, flipping, broadcasting, and zero-copy reshape where the mapping permits it. |

Each engine owns its storage and implements its own shape, indexing, and operation contracts. See the [CPU layout guide](docs/cpu-tensor-layouts.md) for checked reshape, offsets, and write permissions.

## Setup and tests

Install [Pixi](https://pixi.sh) and run these commands from the repository root. The checked-in lockfile pins Mojo 1.1.0 and MAX 26.6.0. The workspace declares Linux x86-64, Linux AArch64, and macOS arm64. The CPU checks for these layout changes ran on Linux x86-64.

```sh
pixi install --locked
pixi run mojo --version
pixi run list-tests
pixi run test-smoke
pixi run test-all
pixi run package
test -s dist/momijo.mojoc
```

Run an individual native test or use it as an API example.

```sh
pixi run mojo run -I src -I . tests/tensor/reshape_in_place.mojo
pixi run mojo run -I src -I . tests/tensor_light/checked_layout_views.mojo
```

The package task produces a compiler-matched `dist/momijo.mojoc` artifact. GPU execution and other platforms are outside the validation reported here. The dataframe, learn, vision, and visual namespaces are reserved.

## Authors and license

Momijo was created by **Morteza Taleblou** and **Mitra Daneshmand**, its co-authors and copyright holders. The source is distributed under the [MIT License](LICENSE).

See [Contributing](CONTRIBUTING.md) for development guidelines.
