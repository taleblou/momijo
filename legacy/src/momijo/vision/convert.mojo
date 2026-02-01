# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision
# File: src/momijo/vision/convert.mojo

from momijo.vision.tensor import Tensor
from momijo.tensor import tensor
from collections.list import List


fn _load_u8(p: UnsafePointer[UInt8], idx: Int) -> UInt8:
    var q = p + idx
    return q.load()

fn _fast_hwc_u8_to_f64(src: image.Tensor, normalize: Bool) -> tensor.Tensor[Float64]:
    var H = src._shape0
    var W = src._shape1
    var C = src._shape2
    var n = H * W * C
    var data = List[Float64]()
    data.reserve(n)
    var i = 0
    while i < n:
        var u: UInt8 = _load_u8(src._data, i)
        var f = Float64(u)
        if normalize:
            f = f / 255.0
        data.append(f)
        i = i + 1
    var shape = List[Int]()
    shape.append(H); shape.append(W); shape.append(C)
    return tensor.Tensor[Float64](data, shape)

fn to_tensor_float64(src: image.Tensor, layout: String = String("HWC"), normalize: Bool = True) -> tensor.Tensor[Float64]:
    if src._ndim == 3 and src._shape2 > 0 and src._stride2 == 1 and layout == String("HWC"):
        return _fast_hwc_u8_to_f64(src, normalize)
    var H = 1; var W = 1; var C = 1
    if src._ndim == 3:
        var h0 = src._shape0; var w0 = src._shape1; var c0 = src._shape2
        if layout == String("CHW"):
            C = c0; H = h0; W = w0
        else:
            H = h0; W = w0; C = c0
    elif src._ndim == 2:
        H = src._shape0; W = src._shape1; C = 1
    else:
        H = src._len; W = 1; C = 1
    var shape = List[Int]()
    shape.append(H); shape.append(W); shape.append(C)
    var out = List[Float64]()
    out.reserve(H * W * C)
    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var ch = 0
            while ch < C:
                var i0 = y; var i1 = x; var i2 = ch
                var elem_off = i0 * src._stride0 + i1 * src._stride1 + i2 * src._stride2
                var u: UInt8 = _load_u8(src._data, elem_off)
                var f = Float64(u)
                if normalize:
                    f = f / 255.0
                out.append(f)
                ch = ch + 1
            x = x + 1
        y = y + 1
    return tensor.Tensor[Float64](out, shape)


fn _fast_hwc_u8_to_f32(src: image.Tensor, normalize: Bool) -> tensor.Tensor[Float32]:
    var H = src._shape0
    var W = src._shape1
    var C = src._shape2
    var n = H * W * C

    var data = List[Float32]()
    data.reserve(n)

    var i = 0
    while i < n:
        var u: UInt8 = _load_u8(src._data, i)
        var f = Float32(u)
        if normalize:
            f = f / Float32(255.0)
        data.append(f)
        i = i + 1

    var shape = List[Int]()
    shape.append(H); shape.append(W); shape.append(C)
    return tensor.Tensor[Float32](data, shape)

fn to_tensor_float32(src: image.Tensor, layout: String = String("HWC"), normalize: Bool = True) -> tensor.Tensor[Float32]:
    # Fast path for tightly-packed HWC u8 images
    if src._ndim == 3 and src._shape2 > 0 and src._stride2 == 1 and layout == String("HWC"):
        return _fast_hwc_u8_to_f32(src, normalize)

    # Determine output H, W, C by the requested layout
    var H = 1
    var W = 1
    var C = 1
    if src._ndim == 3:
        var h0 = src._shape0
        var w0 = src._shape1
        var c0 = src._shape2
        if layout == String("CHW"):
            C = c0; H = h0; W = w0
        else:
            H = h0; W = w0; C = c0
    elif src._ndim == 2:
        H = src._shape0; W = src._shape1; C = 1
    else:
        H = src._len; W = 1; C = 1

    var shape = List[Int]()
    shape.append(H); shape.append(W); shape.append(C)

    var out = List[Float32]()
    out.reserve(H * W * C)

    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var ch = 0
            while ch < C:
                var i0 = y
                var i1 = x
                var i2 = ch
                var elem_off = i0 * src._stride0 + i1 * src._stride1 + i2 * src._stride2
                var u: UInt8 = _load_u8(src._data, elem_off)
                var f = Float32(u)
                if normalize:
                    f = f / Float32(255.0)
                out.append(f)
                ch = ch + 1
            x = x + 1
        y = y + 1

    return tensor.Tensor[Float32](out, shape)
