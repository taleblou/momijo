# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: src/momijo/tensor/blas.mojo

from momijo.tensor.tensor import Tensor

fn _check_sgemm_contract(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32]
) -> Bool:
    # TODO: add real shape/dtype checks once Tensor API is stable
    return True
fn _ensure_contiguous(x: Tensor[Float32]) -> Tensor[Float32]:
    # TODO: if you later add x.is_contiguous()/x.contiguous(), branch here
    return x
fn _zero_out(c: Tensor[Float32]) -> None:
    # TODO: implement c.fill(0.0f32) once available
    pass

# Tiled path placeholder
fn _sgemm_tiled(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32]
) -> Bool:
    # TODO: implement once Tensor supports element access
    return False

# Alpha/Beta-aware tiled placeholder
fn _sgemm_tiled_alpha_beta(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32],
    alpha: Float32,
    beta: Float32
) -> Bool:
    # TODO: implement once Tensor supports element access
    return False

# BLAS FFI placeholder
fn _sgemm_via_blas(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32],
    alpha: Float32 = Float32(1.0),
    beta:  Float32 = Float32(0.0)
) -> Bool:
    return False

# --- public API ---------------------------------------------------------------
fn sgemm(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32],
    alpha: Float32 = Float32(1.0),
    beta:  Float32 = Float32(0.0),
    use_blas: Bool = True
) -> Bool:
    if not _check_sgemm_contract(a, b, out_tensor):
        return False
    if use_blas and _sgemm_via_blas(_ensure_contiguous(a), _ensure_contiguous(b), _ensure_contiguous(out_tensor), alpha, beta):
        return True
    return _sgemm_tiled_alpha_beta(a, b, out_tensor, alpha, beta)