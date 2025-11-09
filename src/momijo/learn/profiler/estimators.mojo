# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.profiler.estimators
# File:         src/momijo/learn/profiler/estimators.mojo
#
# Description:
#   Simple FLOPs/ops estimators for common layers/activations.

@always_inline
fn est_ops_linear(n: Int, fin: Int, fout: Int) -> Float32:
    # Approximate MAC count for y = x @ W^T + b  ~ 2*n*fin*fout
    return Float32(2 * n * fin * fout)

@always_inline
fn est_ops_relu(n: Int, d: Int) -> Float32:
    return Float32(n * d)
