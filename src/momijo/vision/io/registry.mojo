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
# Project: momijo.vision.io
# File: momijo/vision/io/registry.mojo

# MIT License
# Project: momijo.vision.io
# SPDX-License-Identifier: MIT
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# File: src/momijo/vision/io/registry.mojo

# Unified image reader registry.
# Tries PPM, JPEG, PNG based on file extension (case-insensitive).
# Each backend returns (ok, Tensor). In pure-Mojo builds, JPEG/PNG stubs return
# (False, dummy) so pipelines can continue in test mode.

from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType

# Backends (stubs already provided earlier)
from momijo.vision.io.jpeg import read_jpeg, read_jpeg_with_fallback
from momijo.vision.io.png  import read_png,  read_png_with_fallback

 

# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

# -------------------------------------------------------------------------
# Internal: make a small dummy packed HWC u8 tensor (simple gradient).
# Produces a valid Tensor layout using packed_hwc_strides and UInt8 pixels.
# -------------------------------------------------------------------------
fn _make_dummy_u8_hwc(h: Int = 32, w: Int = 32, c: Int = 3) -> Tensor:
    var hh = h
    var ww = w
    var cc = c
    if hh <= 0: hh = 32
    if ww <= 0: ww = 32
    if cc != 1 and cc != 3: cc = 3

    var (s0, s1, s2) = packed_hwc_strides(hh, ww, cc)
    var n = hh * ww * cc
    var buf = UnsafePointer[UInt8].alloc(n)

    var y = 0
    while y < hh:
        var x = 0
        while x < ww:
            var ch = 0
            while ch < cc:
                var v = (y * 7 + x * 11 + ch * 29) % 256
                buf[y * s0 + x * s1 + ch * s2] = UInt8(v)
                ch += 1
            x += 1
        y += 1

    # Construct a Tensor consistent with your project's constructor signature.
    return Tensor(buf, n, hh, ww, cc, s0, s1, s2, DType.UInt8)

fn _to_lower(s: String) -> String:
    var out = String("")
    var i = 0
    var n = s.__len__()
    while i < n:
        var ch = s[i]
        # ASCII tolower
        if ch >= 'A' and ch <= 'Z':
            out = out + String(Char(ord(ch) + 32))
        else:
            out = out + String(ch)
        i += 1
    return out

# -------------------------------------------------------------------
# Optional: PPM file-level reader placeholder
# Replace with a real reader that maps `path -> bytes -> decode_ppm_u8_hwc`.
# -------------------------------------------------------------------
fn read_ppm(path: String) -> (Bool, Tensor):
    # No filesystem in this build; return a dummy to signal "unsupported".
    var dummy = _make_dummy_u8_hwc(32, 32, 3)
    return (False, dummy)

 