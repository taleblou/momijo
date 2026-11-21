<!-- SPDX-FileCopyrightText: 2026 Morteza Taleblou -->
<!-- SPDX-FileCopyrightText: 2026 Mitra Daneshmand -->
<!-- SPDX-License-Identifier: MIT -->

# CPU tensor layouts

## General

`RuntimeTensorShape.try_from_dims` validates every extent and computes a checked element count. Rank zero has one element. Any zero extent makes the count zero, even when a product of the nonzero extents would overflow. Negative extents are invalid even for empty shapes. Coordinates for `TensorF32.get` and `set` are zero-based and must match the runtime rank. An empty shape has no valid coordinates.

`TensorF32.reshape_in_place(dims)` validates the target and then replaces the owner's metadata. The Float32 elements stay in the same allocation and retain their row-major order. The method returns `INVALID_SHAPE` for a negative extent, `OVERFLOW` for an unrepresentable element count, or `SIZE_MISMATCH` when the count differs from the owner's count. A rejection leaves the tensor as it was. Rank changes are permitted, including rank greater than eight. Reshaping an empty tensor requires an empty target. A scalar reshapes to any shape with one element.

General exposes dense owned storage. Borrowed strided views, negative and zero strides, and writable alias mappings are currently unsupported. See the [General native test](../tests/tensor/reshape_in_place.mojo) for an example.

## Light

A host view borrows storage from an owner and carries a fixed-rank layout. Strides and offsets are in Float32 elements. An in-bounds coordinate maps to the layout's origin offset plus the sum of each coordinate times its signed stride. Checked lookups also accept negative coordinates relative to the corresponding axis end. A negative stride walks backwards through storage. An extent greater than one with zero stride repeats the same addresses along that axis.

`TensorLayout.checked` validates dimensions, representable arithmetic, and the full physical address envelope against the backing capacity. A nonempty mapping must fit within the backing elements. An empty mapping has no addresses and may keep an origin at the end of storage. Rank zero has one coordinate, which maps to the origin itself.

A read view can observe repeated addresses. A mutable host view requires a live mutable borrow and a proven injective layout. For up to two axes with extent greater than one, the checked layout's collision test can resolve injectivity even when the strides interleave. With three or more such axes, writeability may be rejected when the checks cannot prove uniqueness. A successful layout check alone says nothing about the relationship between two separate storage borrows.

`slice_axis` accepts a negative step, and `expand_view` can broadcast singleton dimensions. `reshape_view` preserves the C-order sequence of physical addresses when a new affine layout can represent it. It returns a failure status when it cannot represent that sequence. These operations keep the existing storage borrow. The [Light native test](../tests/tensor_light/checked_layout_views.mojo) exercises a reversed slice followed by reshape and broadcast, and checks that interleaved writes reach the expected backing elements.
