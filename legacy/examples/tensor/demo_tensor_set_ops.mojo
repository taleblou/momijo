# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tests
# Module:       tests.test_tensor_set_ops
# File:         examples/Tensor/demo_tensor_set_ops.mojo
#
# Description:
#   Set operations test for momijo.tensor:
#   - union
#   - intersection
#   - difference
#   - symmetric difference (xor)
#   - simple length sanity checks
#   Notes: English-only comments; explicit imports; string-only printing.

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# set operations
# -----------------------------------------------------------------------------
fn np_set_ops() -> None:
    print("\n=== np_set_ops ===")

    # Two integer tensors
    var a = tensor.Tensor([1, 2, 3, 4])
    var b = tensor.Tensor([3, 4, 5])

    # union
    var uni = a.set_union(b)
    print("union: " + uni.__str__())

    # intersection
    var inter = a.set_intersection(b)
    print("intersect: " + inter.__str__())

    # difference (a - b)
    var diff_ab = a.set_difference(b)
    print("setdiff a-b: " + diff_ab.__str__())
    
    # xor (symmetric difference)
    var xor = a.set_xor(b)
    print("setxor: " + xor.__str__())

    # ---- sanity checks ----
    print("len(a): " + String(a.len()) + " | len(b): " + String(b.len()))
    print("len(union): " + String(uni.len()))
    print("len(intersect): " + String(inter.len()))
    print("len(diff a-b): " + String(diff_ab.len()))
    print("len(xor): " + String(xor.len()))

    # Echo inputs/outputs
    print("a: " + a.__str__() + " | b: " + b.__str__())
    print("union: " + uni.__str__())
    print("intersect: " + inter.__str__())
    print("diff a-b: " + diff_ab.__str__())
    print("xor: " + xor.__str__())

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    np_set_ops()
