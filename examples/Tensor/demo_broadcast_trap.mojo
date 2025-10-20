# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_broadcast_trap.mojo
#
# Description:
#   Demonstrates a common broadcasting pitfall and the correct fix using
#   rank alignment (unsqueeze/reshape). Example: (3,1) ⊕ (2,) fails unless
#   we first align ranks to (3,1) ⊕ (1,2) → (3,2).
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
# 26) Broadcasting trap
# -----------------------------------------------------------------------------
fn demo_broadcast_trap() -> None:
    banner("26) BROADCAST TRAP")

    # a has shape (3, 1), b has shape (2,)
    var a = tensor.randn_int([3, 1])
    var b = tensor.randn_int([2])

    print("a shape: " + a.shape().__str__())
    print("b shape: " + b.shape().__str__())
    print("a:\n" + a.__str__())
    print("b:\n" + b.__str__())

    # Attempting a + b should fail in strict broadcasting due to rank mismatch.
    # Check first, then either report or compute.
    var can = a.can_broadcast_with(b, False)
    if not can:
        print("expected broadcast error: shapes " + a.shape().__str__() + " and " + b.shape().__str__())
    else:
        print("a + b:\n" + (a + b).__str__())

    # -------------------------------------------------------------------------
    # Correct fix: align ranks using reshape/unsqueeze
    # b: (2,) -> (1, 2) so that (3,1) + (1,2) broadcasts to (3,2)
    # Use reshape to avoid relying on view semantics.
    # -------------------------------------------------------------------------
    var b_aligned = b.reshape([1, 2])
    var y = a + b_aligned
    print("fixed with b.reshape([1,2]) -> (a + b) shape: " + y.shape().__str__())
    print("fixed with b.reshape([1,2]) -> (a + b):\n" + y.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_broadcast_trap()
