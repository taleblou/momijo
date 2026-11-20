# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT

from ..storage import TensorF32
from ..validation import TensorResult, TensorStatusCode


def sum_all_f32(tensor: TensorF32) -> TensorResult[Float32]:
    """Adds elements in logical order using Float32, starting from +0.0."""
    var total = Float32(0.0)
    for value in tensor._elements_in_logical_order():
        total += value
    return TensorResult[Float32](total)


def mean_all_f32(tensor: TensorF32) -> TensorResult[Float32]:
    """Divides the ordered Float32 sum by the Float32 element count.

    Returns SIZE_MISMATCH for empty input because the mean is undefined.
    """
    var count = tensor.numel()
    if count == 0:
        return TensorResult[Float32](TensorStatusCode.SIZE_MISMATCH)
    var total = sum_all_f32(tensor)
    return TensorResult[Float32](total.take_value() / Float32(count))
