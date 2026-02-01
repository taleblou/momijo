# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_inplace_notes.mojo
#
# Description:
#   In-place ops & grad safety notes — views vs. copies, and out-of-place
#   equivalents. Shows how view semantics affect writes, and why out-of-place
#   ops are often safer for autograd.
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
# 13) In-place notes
# -----------------------------------------------------------------------------
fn demo_inplace_notes() -> None:
    banner("13) IN-PLACE OPS & GRAD SAFETY")

    # Base tensor (Float64 for stable printing)
    var x = tensor.arange_f64(0, 6, 1).reshape([2, 3])
    print("x (base):\n" + x.__str__())

    # View (shares storage): first two columns
    var v = x.slice(1, 0, 2)  # v is a view of x[:, 0:2]
    print("v = x[:, 0:2] (view):\n" + v.__str__())

    # Copy of the same slice (independent storage)
    var c = x.slice(1, 0, 2).clone()
    print("c = clone(x[:, 0:2]) (copy):\n" + c.__str__())

    # -------------------------------------------------------------------------
    # Out-of-place on a view: does NOT modify the base tensor.
    # If you implement true in-place ops (e.g., add_inplace), use that here.
    # -------------------------------------------------------------------------
    var v_add = v.add_scalar(100.0)
    print("after v.add_scalar(100) [out-of-place]:")
    print("v_add (new tensor):\n" + v_add.__str__())
    print("x (base unchanged):\n" + x.__str__())
    print("c (independent copy unchanged):\n" + c.__str__())

    # -------------------------------------------------------------------------
    # Example of writing through a view: fill the first row slice.
    # This does modify the base because row() returns a view.
    # -------------------------------------------------------------------------
    var row0 = x.row(0)  # view of first row
    row0.fill(999.0)     # in-place write through the view
    print("after row0.fill(999): x becomes:\n" + x.__str__())

    # Out-of-place equivalent: returns a new tensor; base x is unchanged
    var z = x.add_scalar(5.0)
    print("z = x.add_scalar(5) (out-of-place):\n" + z.__str__())
    print("x unchanged by out-of-place add:\n" + x.__str__())

    # Clamp (out-of-place variant shown here)
    var x_clamped = x.clamped(-50.0, 200.0)
    print("x.clamped(-50, 200):\n" + x_clamped.__str__())

    # Type-changing ops are out-of-place by design
    var xi = x.to_int()
    print("to_int (out-of-place cast):\n" + xi.__str__())

    # -------------------------------------------------------------------------
    # Notes on grad safety (placeholder — no autograd required to run this)
    # -------------------------------------------------------------------------
    print("\n[Notes]")
    print("- Writes through views modify all aliases pointing to the same storage.")
    print("- Prefer out-of-place ops during backward unless your API guarantees safety.")
    print("- Dtype/device changes are typically out-of-place; plan memory accordingly.")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_inplace_notes()
