# Project:      Momijo
# Module:       learn.nn.layers
# File:         nn/layers.mojo
# Path:         src/momijo/learn/nn/layers.mojo
#
# Description:  Core neural network layers for Momijo Learn. This file defines
#               the public-facing layer types (Linear, Conv2d, BatchNorm2d,
#               Dropout, Flatten) with stable constructors and forward() APIs.
#               The math is backend-agnostic; wire real tensor ops later.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Types: Linear, Conv2d, BatchNorm2d, Dropout, Flatten
#   - Public API: forward(x), reset_parameters(), __str__()
#   - Parameters are declared structurally; connect to momijo.tensor weights later.
#   - No wildcard imports; constructors use `out self`; printing uses __str__ only.

from collections.list import List
from momijo.learn.nn.module import Module
from momijo.learn.nn.init import (
    kaiming_uniform,
    xavier_uniform,
)
from momijo.learn.nn.regularizers import (
    l1_penalty,
    l2_penalty,
)

# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

struct Linear(Module):
    var in_features: Int
    var out_features: Int
    var bias: Bool

    # Placeholder parameter storage (replace with Tensor later)
    var weight_shape_mn: List[Int]     # [out_features, in_features]
    var bias_shape_n: List[Int]        # [out_features]

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # Shapes only; connect to real tensors/alloc later
        self.weight_shape_mn = List[Int]()
        self.weight_shape_mn.push_back(out_features)
        self.weight_shape_mn.push_back(in_features)
        self.bias_shape_n = List[Int]()
        if bias:
            self.bias_shape_n.push_back(out_features)
        self.reset_parameters()

    fn reset_parameters(mut self):
        # Wire real initialization once weight/bias tensors exist
        # xavier_uniform([out_features, in_features]); bias zeros
        pass

    fn forward(self, x):
        # Replace with: x @ W.T + b
        return x

    fn __str__(self) -> String:
        var s = String("Linear(")
        s += String("in_features=") + String(self.in_features)
        s += String(", out_features=") + String(self.out_features)
        s += String(", bias=") + (String("True") if self.bias else String("False"))
        s += String(")")
        return s


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

struct Conv2d(Module):
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var groups: Int
    var bias: Bool

    # Placeholder parameter shapes (replace with Tensor later)
    # weight: [out_channels, in_channels/groups, k, k]
    var weight_shape: List[Int]
    # bias: [out_channels]
    var bias_shape: List[Int]

    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = True
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight_shape = List[Int]()
        self.weight_shape.push_back(out_channels)
        # in_channels per group
        var icg = in_channels // groups
        self.weight_shape.push_back(icg)
        self.weight_shape.push_back(kernel_size)
        self.weight_shape.push_back(kernel_size)

        self.bias_shape = List[Int]()
        if bias:
            self.bias_shape.push_back(out_channels)

        self.reset_parameters()

    fn reset_parameters(mut self):
        # He/Kaiming init for conv weights; bias zeros
        # kaiming_uniform([oc, icg, k, k])
        pass

    fn forward(self, x):
        # Replace with: conv2d(x, weight, bias, stride, padding, dilation, groups)
        return x

    fn __str__(self) -> String:
        var s = String("Conv2d(")
        s += String("in_channels=") + String(self.in_channels)
        s += String(", out_channels=") + String(self.out_channels)
        s += String(", kernel_size=") + String(self.kernel_size)
        s += String(", stride=") + String(self.stride)
        s += String(", padding=") + String(self.padding)
        s += String(", dilation=") + String(self.dilation)
        s += String(", groups=") + String(self.groups)
        s += String(", bias=") + (String("True") if self.bias else String("False"))
        s += String(")")
        return s


# ---------------------------------------------------------------------------
# BatchNorm2d
# ---------------------------------------------------------------------------

struct BatchNorm2d(Module):
    var num_features: Int
    var eps: Float64
    var momentum: Float64
    var affine: Bool
    var track_running_stats: Bool

    # Placeholder parameter/buffer shapes
    # weight(gamma), bias(beta): [num_features] if affine
    # running_mean, running_var: [num_features] if tracking stats
    var weight_shape: List[Int]
    var bias_shape: List[Int]
    var running_mean_shape: List[Int]
    var running_var_shape: List[Int]

    fn __init__(
        out self,
        num_features: Int,
        eps: Float64 = 1e-5,
        momentum: Float64 = 0.1,
        affine: Bool = True,
        track_running_stats: Bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.weight_shape = List[Int]()
        self.bias_shape = List[Int]()
        self.running_mean_shape = List[Int]()
        self.running_var_shape = List[Int]()

        if affine:
            self.weight_shape.push_back(num_features)
            self.bias_shape.push_back(num_features)
        if track_running_stats:
            self.running_mean_shape.push_back(num_features)
            self.running_var_shape.push_back(num_features)

        self.reset_parameters()

    fn reset_parameters(mut self):
        # gamma = 1, beta = 0; running_mean=0, running_var=1
        pass

    fn forward(self, x, training: Bool = True):
        # Replace with true BN: normalize channel-wise using running stats in eval
        # and batch stats in train; update running stats if tracking.
        return x

    fn __str__(self) -> String:
        var s = String("BatchNorm2d(")
        s += String("num_features=") + String(self.num_features)
        s += String(", eps=") + String(self.eps)
        s += String(", momentum=") + String(self.momentum)
        s += String(", affine=") + (String("True") if self.affine else String("False"))
        s += String(", track_running_stats=") + (String("True") if self.track_running_stats else String("False"))
        s += String(")")
        return s


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

struct Dropout(Module):
    var p: Float64
    var inplace: Bool

    fn __init__(out self, p: Float64 = 0.5, inplace: Bool = False):
        self.p = p
        self.inplace = inplace

    fn forward(self, x, training: Bool = True):
        # If training: randomly zero-out elements with prob p; else return x
        # Placeholder returns x as-is.
        return x

    fn __str__(self) -> String:
        var s = String("Dropout(")
        s += String("p=") + String(self.p)
        s += String(", inplace=") + (String("True") if self.inplace else String("False"))
        s += String(")")
        return s


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

struct Flatten(Module):
    var start_dim: Int
    var end_dim: Int

    fn __init__(out self, start_dim: Int = 1, end_dim: Int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    fn forward(self, x):
        # Replace with real reshape that flattens dims [start_dim..end_dim]
        return x

    fn __str__(self) -> String:
        var s = String("Flatten(")
        s += String("start_dim=") + String(self.start_dim)
        s += String(", end_dim=") + String(self.end_dim)
        s += String(")")
        return s
