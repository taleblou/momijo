# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
# Source: tests/tensor/reshape_in_place.mojo
# Module: tests.tensor.reshape_in_place

from std.testing import assert_equal, assert_true

from momijo.tensor import (
    RuntimeTensorShape,
    TensorF32,
    TensorStatusCode,
    ones_f32,
    sum_all_f32,
)


def check_row_major_order() raises:
    var shape = RuntimeTensorShape.try_from_dims([2, 3])
    assert_true(shape.is_ok())
    var made = ones_f32(shape.take_value())
    assert_true(made.is_ok())
    var tensor = made.take_value()
    for i in range(2):
        for j in range(3):
            assert_true(tensor.set([i, j], Float32(i * 3 + j + 1)) == TensorStatusCode.OK)

    var dims = List[Int](length=12, fill=1)
    dims[7] = 3
    dims[11] = 2
    assert_true(tensor.reshape_in_place(dims) == TensorStatusCode.OK)
    assert_equal(tensor.rank(), 12)
    assert_equal(tensor.numel(), 6)
    var coordinate = List[Int](length=12, fill=0)
    for i in range(3):
        for j in range(2):
            coordinate[7] = i
            coordinate[11] = j
            var value = tensor.get(coordinate)
            assert_true(value.is_ok())
            assert_equal(value.take_value(), Float32(i * 2 + j + 1))
    assert_true(tensor.set(coordinate, 9.0) == TensorStatusCode.OK)
    var sum = sum_all_f32(tensor)
    assert_true(sum.is_ok())
    assert_equal(sum.take_value(), Float32(24.0))

    assert_true(tensor.reshape_in_place([5]) == TensorStatusCode.SIZE_MISMATCH)
    assert_true(tensor.reshape_in_place([0, -1]) == TensorStatusCode.INVALID_SHAPE)
    assert_true(tensor.reshape_in_place([Int.MAX, 2]) == TensorStatusCode.OVERFLOW)
    assert_equal(tensor.rank(), 12)
    var last = tensor.get(coordinate)
    assert_true(last.is_ok())
    assert_equal(last.take_value(), Float32(9.0))


def check_scalar_and_empty() raises:
    var scalar = TensorF32(4.0)
    var no_axes = List[Int]()
    assert_equal(scalar.rank(), 0)
    assert_true(scalar.reshape_in_place([1, 1]) == TensorStatusCode.OK)
    assert_true(scalar.reshape_in_place(no_axes) == TensorStatusCode.OK)
    var value = scalar.get(no_axes)
    assert_true(value.is_ok())
    assert_equal(value.take_value(), Float32(4.0))

    var shape = RuntimeTensorShape.try_from_dims([0, 1])
    assert_true(shape.is_ok())
    var made = ones_f32(shape.take_value())
    assert_true(made.is_ok())
    var empty = made.take_value()
    # The trailing zero makes the product zero even if a prefix overflows.
    assert_true(empty.reshape_in_place([Int.MAX, Int.MAX, 0]) == TensorStatusCode.OK)
    assert_equal(empty.numel(), 0)
    assert_equal(empty.rank(), 3)
    assert_true(empty.get([0, 0, 0]).status() == TensorStatusCode.OUT_OF_BOUNDS)
    assert_true(empty.reshape_in_place(no_axes) == TensorStatusCode.SIZE_MISMATCH)


def main() raises:
    check_row_major_order()
    check_scalar_and_empty()
