# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tests
# Module:       tests.test_tensor_indexing
# File:         examples/Tensor/demo_tensor_indexing.mojo
#
# Description:
#   Advanced indexing, gather, and plane/broadcast write tests for momijo.tensor.
 

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# more indexing
# -----------------------------------------------------------------------------
fn np_more_indexing() -> None:
    print("\n=== np_more_indexing ===")

    # a has shape [2, 3, 4] with values 0..23
    var a = tensor.arange(0, 24).reshape([2, 3, 4])
    print("a: " + a.__str__())
    print("a shape: " + a.shape().__str__())

    # Indices for gather over axis=0 after slicing dim0
    var idx = tensor.Tensor([0, 2, 1])

    # Slice along dim-0: a0 should have shape [3, 4]
    var a0 = tensor.slice_dim0(a, 0)
    print("a0 (a[0, ...]): " + a0.__str__())
    print("a0 shape: " + a0.shape().__str__())

    # Gather along axis=0 using idx => picks rows 0,2,1 from a0
    var picked = tensor.gather(a0, 0, idx)
    print("picked (gather a0 over axis=0 with [0,2,1]): " + picked.__str__())
    print("picked shape: " + picked.shape().__str__())

    # Copy 'a' before plane write
    var b = a.copy()

    # Broadcast write into the plane (last dimension index = 0) across the middle dim
    # write_plane(tensor, axis, index, value)
    # Here: axis=2 (last axis), index=0, value shape [2,1] -> broadcast to [2,3]
    var to_write = tensor.Tensor([100, 200]).reshape([2, 1])
    tensor.write_plane(b, 2, 0, to_write)

    # Extract plane to verify: plane(b, axis, index) -> shape [2,3]
    var p = tensor.plane(b, 2, 0)
    print("broadcast assign b[:,:,0]: " + p.__str__())
    print("plane shape (should be [2,3]): " + p.shape().__str__())

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    np_more_indexing()
