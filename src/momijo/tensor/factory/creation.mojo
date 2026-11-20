# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT

from ..shape.shape import RuntimeTensorShape
from ..storage import TensorF32
from ..validation import TensorResult


def ones_f32(var shape: RuntimeTensorShape) -> TensorResult[TensorF32]:
    """Takes ownership of the shape and initializes every element to 1.0.

    Returns OVERFLOW if the Float32 buffer size exceeds Int.MAX bytes.
    For representable sizes, memory exhaustion follows Mojo List's behavior.
    """
    return TensorF32._filled(shape^, Float32(1.0))
