# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/layernorm.mojo
# Description: LayerNorm over last-dimension (2D: [N,C]).

from momijo.tensor import tensor

struct LayerNorm1d:
    var num_features: Int
    var gamma: tensor.Tensor[Float32]
    var beta: tensor.Tensor[Float32]
    var eps: Float32

    fn __init__(out self, num_features: Int, eps: Float32 = 1e-5):
        self.num_features = num_features
        self.gamma = tensor.ones([num_features])
        self.beta = tensor.zeros([num_features])
        self.eps = eps

    fn __copyinit__(out self, other: Self):
        self.num_features = other.num_features
        self.gamma = other.gamma
        self.beta = other.beta
        self.eps = other.eps

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # x: [N,C], normalize across C for each sample
        var N = x.shape()[0]
        var C = x.shape()[1]
        var ones_c = tensor.ones([C,1])
        var mean = tensor.matmul(x, ones_c) / Float32(C)   # [N,1]
        var xm = x - mean
        var varv = tensor.matmul(xm * xm, ones_c) / Float32(C)  # [N,1]
        var invstd = tensor.reciprocal(tensor.sqrt(varv + self.eps))
        var y = xm * invstd   # [N,C] with broadcasting on second dim
        return y * self.gamma + self.beta
