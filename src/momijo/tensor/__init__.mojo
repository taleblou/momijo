# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
# Source: src/momijo/tensor/__init__.mojo
# Module: momijo.tensor

"""General CPU tensors with runtime rank and owned, row-major storage."""

from .validation import TensorResult, TensorStatusCode
from .shape.shape import RuntimeTensorShape
from .storage import TensorF32
from .factory.creation import ones_f32
from .ops.reduction import mean_all_f32, sum_all_f32
