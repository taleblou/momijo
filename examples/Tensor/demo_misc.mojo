# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_misc.mojo
#
# Description:
#   Miscellaneous tensor utilities demo:
#   - Converting tensors to nested lists for pretty-printing
#   - Basic metadata (numel / ndim / shape)
#   - Scalar construction
#   - In-place fill and clamp examples
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# Pretty-printers for List[Float64] and List[List[Float64]]
# -----------------------------------------------------------------------------
@always_inline
fn list1d_f64_str(xs: List[Float64]) -> String:
    var s = String("[")
    var n = len(xs)
    var i = 0
    while i < n:
        s = s + String(xs[i])
        if i + 1 < n:
            s = s + ", "
        i += 1
    s = s + "]"
    return s

@always_inline
fn list2d_f64_str(xss: List[List[Float64]]) -> String:
    var s = String("[")
    var n = len(xss)
    var i = 0
    while i < n:
        s = s + list1d_f64_str(xss[i])
        if i + 1 < n:
            s = s + ", "
        i += 1
    s = s + "]"
    return s

# -----------------------------------------------------------------------------
# 20) Misc utilities
# -----------------------------------------------------------------------------
fn demo_misc() -> None:
    banner("20) MISC UTILS")

    var x = tensor.arange_f64(0, 6, 1).reshape([2, 3])
    print("x:\n" + x.__str__())

    # Convert to a nested list for display
    print("tolist(): " + list2d_f64_str(x.to_list()))

    # Basic metadata
    print(
        "numel: " + String(x.numel())
        + " | ndim: " + String(x.ndim())
        + " | shape: " + x.shape().__str__()
    )

    # Scalar tensor example
    var s = tensor.scalar_f64(3.14159)
    print("scalar:\n" + s.__str__())

    # In-place fill and clamp (demonstrates mutating APIs)
    var y = x.clone()
    y.fill(42.0)            # fill in-place
    y.clamp(None, 10.0)     # clamp in-place; keeps values <= 10.0
    print("fill_ then clamp_:\n" + y.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_misc()
