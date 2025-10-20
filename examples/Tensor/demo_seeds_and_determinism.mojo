# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_seeds_and_determinism.mojo
#
# Description:
#   Demo for seeds and determinism (placeholder). Shows reproducible draws
#   when using the same seed vs. different results for a different seed.
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
# 23) Seeds & determinism
# -----------------------------------------------------------------------------
fn demo_seeds_and_determinism() -> None:
    banner("23) SEEDS & DETERMINISM")

    # Draw three 1D normal tensors:
    # - a1 and a2 use the same seed → expected equal values
    # - a3 uses a different seed → expected different values
    var a1 = tensor.randn_f64([3], 123)
    var a2 = tensor.randn_f64([3], 123)
    var a3 = tensor.randn_f64([3], 3)

    print("a1:\n" + a1.__str__())
    print("a2:\n" + a2.__str__())
    print("a3:\n" + a3.__str__())

    # Placeholder notes — exact determinism depends on the RNG implementation.
    print("same(a1, a2) with same seed: " + (a1.equal(a2).__str__()))
    print("same(a1, a3) with different seeds: " + (a1.equal(a3).__str__()))
    print("Note: determinism requires fully seedable RNG across all kernels/devices.")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_seeds_and_determinism()
