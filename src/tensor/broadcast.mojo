# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Minimal broadcasting for elementwise add/mul with one dimension 1 support (2D demo)

from momijo.tensor.tensor import Tensor

fn add_broadcast_2d(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    # supports cases: (m,n)+(1,n) or (m,1)+(m,n) or exact match
    assert(len(a.shape) == 2 and len(b.shape) == 2, "Only 2D supported")
    let m = a.shape[0]
    let n = a.shape[1]
    assert((b.shape[0] == m or b.shape[0] == 1) and (b.shape[1] == n or b.shape[1] == 1), "Shapes not broadcastable")
    var out = Tensor[Float64](shape=[m, n], fill=0.0)
    for i in range(0, m):
        for j in range(0, n):
            let av = a.get([i, j])
            let bv = b.get([0 if b.shape[0]==1 else i, 0 if b.shape[1]==1 else j])
            out.set([i, j], av + bv)
    return out
