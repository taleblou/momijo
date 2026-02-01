# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.vision.transforms.image
# File:         src/momijo/vision/transforms/transforms.mojo
#
# Description:
#   Image-focused transforms on Momijo tensors (Float32).
#   - to_tensor: HW/HWC(0..255) -> CHW(0..1)
#   - normalize_chw: per-channel normalization
#   - random_affine_placeholder: no-op placeholder (wire real warp when available)
#   - ImageCompose: configurable pipeline (affine/normalize/resize hook)
#
# Notes:
#   * Expects tensor ops: permute, mul_scalar, add_scalar, slice, assign_slice, reshape
#   * Works for grayscale or RGB. For grayscale, HW or HWC(1) are both handled.

from collections.list import List
from pathlib import Path
from momijo.tensor import tensor
from momijo.vision.transforms.compose import Compose

# ----------------------------- Low-level transforms ---------------------------

@always_inline
fn to_tensor(img: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    # Accept HW or HWC in 0..255; return CHW in 0..1
    var a = img.copy()
    var shp = a.shape()
    if len(shp) == 2:
        # HW -> HWC(1)
        a = a.reshape([shp[0], shp[1], 1])
        shp = a.shape()
    # 0..255 -> 0..1
    a = a.mul_scalar(1.0 / 255.0)
    # HWC -> CHW
    return a.permute([2, 0, 1])

@always_inline
fn normalize_chw(x: tensor.Tensor[Float32], mean: List[Float32], std: List[Float32]) -> tensor.Tensor[Float32]:
    var out = x.copy()            # [C,H,W]
    var shp = out.shape()
    var c = shp[0]
    var h = shp[1]
    var w = shp[2]
    var i = 0
    while i < c:
        var mi = (mean[i] if i < len(mean) else 0.0)
        var si = (std[i]  if i < len(std)  else 1.0)
        # slice channel i
        var xi = out.slice([i, 0, 0], [i + 1, h, w]).copy()
        xi = xi.add_scalar(-mi)
        xi = xi.mul_scalar(1.0 / si)
        out = out.assign_slice([i, 0, 0], xi)
        i += 1
    return out.copy()

# Hook: replace with real affine warp when ready (vision.geometry.affine_warp)
fn random_affine_placeholder(x: tensor.Tensor[Float32],
                             degrees: Float32,
                             tx: Float32,
                             ty: Float32,
                             smin: Float32,
                             smax: Float32) -> tensor.Tensor[Float32]:
    # TODO: Plug-in real affine (rotation+translate+scale) here.
    return x

# Optional: resize hook (no-op for MNIST=28). Keep placeholder for future wiring.
fn maybe_resize_placeholder(x: tensor.Tensor[Float32], img_size: Int) -> tensor.Tensor[Float32]:
    # If you already have resize bilinear: call it here.
    # For now: do nothing to avoid extra deps.
    return x.copy()


# ----------------------------- Preset builders --------------------------------



fn build_transforms(img_size: Int = 28, augment: Bool = False) -> Compose[tensor.Tensor[Float32]]:
    var t = Compose[tensor.Tensor[Float32]]()
    if augment:
        t = t.add_random_affine(10.0, 0.05, 0.05, 0.95, 1.05)
    var mean = List[Float32](); mean.append(0.1307)
    var std  = List[Float32](); std.append(0.3081)
    t = t.add_normalize(mean, std)
    return t.copy()
