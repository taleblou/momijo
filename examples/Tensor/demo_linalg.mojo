# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_linalg.mojo
#
# Description:
#   Demo for linear algebra: matmul (mv/mm), solve, inv, QR, SVD, and Cholesky.
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
# 8) Linear Algebra
# -----------------------------------------------------------------------------
fn demo_linalg() -> None:
    banner("8) LINEAR ALGEBRA")

    # Random square matrix and vectors
    var a = tensor.randn_f64([3, 3])
    var b = tensor.randn_f64([3])

    # Matrix-vector and matrix-matrix products
    print("A@b (mv): " + a.matmul_vec(b).__str__())

    var B = tensor.randn_f64([3, 3])
    print("A@B (mm):\n" + a.matmul(B).__str__())

    # Solve Ax = b
    var x_sol = a.solve(b)
    print("solve Ax=b:\n" + x_sol.__str__())

    # Check with inv(A) @ b
    print("inv(A) * b (check): " + a.inv().matmul_vec(b).__str__())

    # QR decomposition
    var qr_pair = a.qr()
    var Q = qr_pair[0].copy()
    var R = qr_pair[1].copy()
    print("QR shapes: " + Q.shape().__str__() + " " + R.shape().__str__())

    # SVD decomposition
    var svd_triplet = a.svd()
    var U  = svd_triplet[0].copy()
    var S  = svd_triplet[1].copy()
    var Vh = svd_triplet[2].copy()
    print("SVD shapes: " + U.shape().__str__() + " " + S.shape().__str__() + " " + Vh.shape().__str__())

    # Build a Symmetric Positive Definite (SPD) matrix: M M^T + eps*I
    var M = tensor.randn_f64([3, 3])
    var I = tensor.eye_f64(3)
    var SPD = M.matmul(M.transpose([1, 0])).add(I.mul_scalar(1e-3))

    # Cholesky factorization (lower-triangular L with SPD = L L^T)
    var L = SPD.cholesky()
    print("Cholesky L:\n" + L.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_linalg()
