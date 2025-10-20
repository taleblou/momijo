# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tests
# Module:       tests.test_tensor_linalg
# File:         examples/Tensor/demo_tensor_linalg.mojo
#
# Description:
#   Advanced linear algebra tests for momijo.tensor:
#   - matrix-vector and matrix-matrix products
#   - tensordot with axis
#   - column-wise reductions 

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# advanced linalg
# -----------------------------------------------------------------------------
fn np_linalg_advanced() -> None:
    print("\n=== np_linalg_advanced ===")

    # A = reshape(1..9) into 3x3
    var A = tensor.arange(1, 10).reshape([3, 3])
    var x = tensor.Tensor([1.0, 0.5, -1.0])

    print("x: " + x.__str__())

    # 1) Matrix-vector product
    var Ax = A.matmul(x)
    print("A@x: " + Ax.__str__())

    # 2) Tensordot with axis=1
    #    Often yields a 3x3 result related to A @ A^T or A^T @ A depending on layout.
    var td = A.tensordot(A, axis=1)
    print("tensordot shape: " + td.shape().__str__())

    # 3) Matrix-matrix product (A @ A)
    var mm = A.matmul(A)
    print("einsum mm: " + mm.__str__())          # label kept as requested

    # 4) Column-wise sum (akin to einsum over rows)
    var colsum = A.sum(axis=0)
    print("einsum colsum: " + colsum.__str__())  # label kept as requested

    # Lightweight sanity prints
    print("A shape: " + A.shape().__str__())
    print("x shape: " + x.shape().__str__())
    print("A@x shape: " + Ax.shape().__str__())
    print("mm shape: " + mm.shape().__str__())

    # Optional coverage (if available in your API)
    print("row sums: " + A.sum(axis=1).__str__())
    print("row means: " + A.mean(axis=1).__str__())

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    np_linalg_advanced()
