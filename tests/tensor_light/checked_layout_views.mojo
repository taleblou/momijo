# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
# Source: tests/tensor_light/checked_layout_views.mojo
# Module: tests.tensor_light.checked_layout_views

from std.testing import assert_equal, assert_true

from momijo.tensor_light.alias.writeability import validate_writable
from momijo.tensor_light.factory.creation import zeros
from momijo.tensor_light.indexing.slice import SliceSpec
from momijo.tensor_light.internal.status import TensorStatus
from momijo.tensor_light.layout.contiguity import InternalOverlapKind
from momijo.tensor_light.layout.layout import TensorLayout
from momijo.tensor_light.shape.shape import TensorShape
from momijo.tensor_light.shape.shape_storage import StaticRankArray


def pair(first: Int, second: Int) -> StaticRankArray[2]:
    var result = StaticRankArray[2](first)
    result.set(1, second)
    return result


def check_interleaved_strides() raises:
    var owner_shape = TensorShape[1].checked(StaticRankArray[1](8))
    assert_true(owner_shape.is_ok())
    var made = zeros[1](owner_shape.payload.value())
    assert_true(made.is_ok())
    var owner = made.payload.take()
    var shape = TensorShape[2].checked(pair(3, 2))
    assert_true(shape.is_ok())
    var strides = pair(-2, 3)
    var addresses: List[Int] = [4, 7, 2, 5, 0, 3]
    var mapped = TensorLayout[2].checked(shape.payload.value(), strides, 4, 8)
    assert_true(mapped.is_ok())
    assert_true(validate_writable(mapped.payload.value(), 8) == TensorStatus.OK,
                "interleaving alone must not rule out unique writes")
    var writable = owner.mut_view()
    assert_true(writable.is_ok())
    var base = writable.payload.take()
    var selected = base.as_strided_mut_view[2](shape.payload.value(), strides, 4)
    assert_true(selected.is_ok())
    var view = selected.payload.take()
    for i in range(3):
        for j in range(2):
            var linear = i * 2 + j
            var coordinate = pair(i, j)
            var address = mapped.payload.value().physical_offset(coordinate)
            assert_true(address.is_ok())
            assert_equal(address.payload.value(), addresses[linear])
            assert_true(view.set(coordinate, Float32(linear + 1)) == TensorStatus.OK)
    # Independently observe the writes from the original dense mapping.
    var expected: List[Float32] = [5.0, 0.0, 3.0, 6.0, 1.0, 4.0, 0.0, 2.0]
    for i in range(8):
        var readback = base.get(StaticRankArray[1](i))
        assert_true(readback.is_ok())
        assert_equal(readback.payload.value(), expected[i])
    var collision = TensorLayout[2].checked(shape.payload.value(), pair(-2, 4), 4, 9)
    assert_true(collision.is_ok())
    assert_true(collision.payload.value().overlap == InternalOverlapKind.DEFINITE)
    assert_true(validate_writable(collision.payload.value(), 9) == TensorStatus.OVERLAPPING)
    # There is no pairwise collision here, but 2 + 3 and 5 address the same cell.
    var three_shape = TensorShape[3].checked(StaticRankArray[3](2))
    assert_true(three_shape.is_ok())
    var three_strides = StaticRankArray[3](2)
    three_strides.set(1, 3)
    three_strides.set(2, 5)
    var three = TensorLayout[3].checked(three_shape.payload.value(), three_strides, 0, 11)
    assert_true(three.is_ok())
    assert_true(three.payload.value().overlap == InternalOverlapKind.UNKNOWN)
    assert_true(validate_writable(three.payload.value(), 11) == TensorStatus.OVERLAPPING)


def check_reverse_and_broadcast() raises:
    var shape = TensorShape[1].checked(StaticRankArray[1](3))
    assert_true(shape.is_ok())
    var made = zeros[1](shape.payload.value())
    assert_true(made.is_ok())
    var owner = made.payload.take()
    for i in range(3):
        assert_true(owner.set(StaticRankArray[1](i), Float32(i + 1)) == TensorStatus.OK)

    var reversed_result = owner.slice_axis(0, SliceSpec(None, None, -1))
    assert_true(reversed_result.is_ok())
    var reversed = reversed_result.payload.take()
    var target = TensorShape[2].checked(pair(1, 3))
    assert_true(target.is_ok())
    var reshaped_result = reversed.reshape_view[2](target.payload.value())
    assert_true(reshaped_result.is_ok())
    var reshaped = reshaped_result.payload.take()
    var expanded_shape = TensorShape[2].checked(pair(2, 3))
    assert_true(expanded_shape.is_ok())
    var expanded_result = reshaped.expand_view[2](expanded_shape.payload.value())
    assert_true(expanded_result.is_ok())
    var expanded = expanded_result.payload.take()
    var layout = expanded.checked_layout()
    assert_true(layout.is_ok())
    assert_true(validate_writable(layout.payload.value(), 3) == TensorStatus.OVERLAPPING)
    var flat_shape = TensorShape[1].checked(StaticRankArray[1](6))
    assert_true(flat_shape.is_ok())
    assert_true(expanded.reshape_view[1](flat_shape.payload.value()).status
                == TensorStatus.NOT_REPRESENTABLE)
    var transposed_result = expanded.transpose(0, 1)
    assert_true(transposed_result.is_ok())
    var transposed = transposed_result.payload.take()
    var permuted_result = transposed.permute(pair(1, 0))
    assert_true(permuted_result.is_ok())
    var permuted = permuted_result.payload.take()
    var permuted_layout = permuted.checked_layout()
    assert_true(permuted_layout.is_ok())
    for i in range(2):
        for j in range(3):
            var coordinate = pair(i, j)
            var value = expanded.get(coordinate)
            assert_true(value.is_ok())
            assert_equal(value.payload.value(), Float32(3 - j))
            var address = layout.payload.value().physical_offset(coordinate)
            assert_true(address.is_ok())
            assert_equal(address.payload.value(), 2 - j)
            var permuted_address = permuted_layout.payload.value().physical_offset(coordinate)
            assert_true(permuted_address.is_ok())
            assert_equal(permuted_address.payload.value(), 2 - j)
    var mutable_result = owner.mut_view()
    assert_true(mutable_result.is_ok())
    var mutable = mutable_result.payload.take()
    assert_true(mutable.expand_mut_view[2](expanded_shape.payload.value()).status
                == TensorStatus.OVERLAPPING)


def check_layout_bounds() raises:
    var scalar = TensorShape[0].checked(StaticRankArray[0](0))
    assert_true(scalar.is_ok())
    var scalar_layout = TensorLayout[0].checked(
        scalar.payload.value(), StaticRankArray[0](0), 2, 3,
    )
    assert_true(scalar_layout.is_ok())
    var scalar_address = scalar_layout.payload.value().physical_offset(StaticRankArray[0](0))
    assert_true(scalar_address.is_ok())
    assert_equal(scalar_address.payload.value(), 2)

    var empty = TensorShape[2].checked(pair(0, Int.MAX))
    assert_true(empty.is_ok())
    var empty_layout = TensorLayout[2].checked(
        empty.payload.value(), pair(Int.MIN, Int.MAX), 3, 3,
    )
    assert_true(empty_layout.is_ok())
    assert_equal(empty_layout.payload.value().span_size(), 0)
    assert_true(empty_layout.payload.value().physical_offset(pair(0, 0)).status
                == TensorStatus.OUT_OF_BOUNDS)
    assert_true(TensorLayout[2].checked(empty.payload.value(), pair(0, 0), 4, 3).status
                == TensorStatus.OUT_OF_BOUNDS)

    var shape = TensorShape[2].checked(pair(3, 2))
    assert_true(shape.is_ok())
    assert_true(TensorLayout[2].checked(shape.payload.value(), pair(Int.MIN, 1), 0, 3).status
                == TensorStatus.OVERFLOW)
    assert_true(TensorLayout[2].checked(shape.payload.value(), pair(-2, 1), 3, 8).status
                == TensorStatus.OUT_OF_BOUNDS)


def main() raises:
    check_interleaved_strides()
    check_reverse_and_broadcast()
    check_layout_bounds()
