# Project:      Momijo
# Module:       src.momijo.vision.ir.fusion
# File:         fusion.mojo
# Path:         src/momijo/vision/ir/fusion.mojo
#
# Description:  src.momijo.vision.ir.fusion — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: BlendMode
#   - Key functions: __init__, NORMAL, ADD, MULTIPLY, SCREEN, DARKEN, LIGHTEN, __eq__ ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


struct BlendMode(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn NORMAL()   -> BlendMode: return BlendMode(0)
    @staticmethod fn ADD()      -> BlendMode: return BlendMode(1)
    @staticmethod fn MULTIPLY() -> BlendMode: return BlendMode(2)
    @staticmethod fn SCREEN()   -> BlendMode: return BlendMode(3)
    @staticmethod fn DARKEN()   -> BlendMode: return BlendMode(4)
    @staticmethod fn LIGHTEN()  -> BlendMode: return BlendMode(5)
fn __eq__(self, other: BlendMode) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("NORMAL")
        if self.id == 1: return String("ADD")
        if self.id == 2: return String("MULTIPLY")
        if self.id == 3: return String("SCREEN")
        if self.id == 4: return String("DARKEN")
        if self.id == 5: return String("LIGHTEN")
        return String("UNKNOWN")

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn _alloc_u8(n: Int) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var i = 0
    while i < n:
        out.append(0)
        i += 1
    return out

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@staticmethod
fn _max(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

# -------------------------
# Core blending (per-channel) for same-size images

# -------------------------
@staticmethod
fn blend_same_size_u8_hwc(mode: BlendMode, h: Int, w: Int, c: Int,
                          A: List[UInt8], B: List[UInt8], opacity: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    var t = opacity
    if t < 0: t = 0
    if t > 255: t = 255
    var inv_t = 255 - t

    var out = _alloc_u8(h * w * c)

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var chn = 0
            while chn < c:
                var idx = _offset(w, c, x, y, chn)
                var a = Int(A[idx])
                var b = Int(B[idx])
                var base = 0

                if mode == BlendMode.NORMAL():
                    base = b
                elif mode == BlendMode.ADD():
                    var s = a + b
                    if s > 255: s = 255
                    base = s
                elif mode == BlendMode.MULTIPLY():
                    base = (a * b + 127) // 255
                elif mode == BlendMode.SCREEN():
                    # screen(a,b) = 1 - (1-a)*(1-b)
                    base = 255 - (( (255 - a) * (255 - b) + 127) // 255)
                elif mode == BlendMode.DARKEN():
                    base = _min(a, b)
                elif mode == BlendMode.LIGHTEN():
                    base = _max(a, b)
                else:
                    base = b

                var outv = (inv_t * a + t * base + 127) // 255
                out[idx] = _clamp_u8(outv)
                chn += 1
            x += 1
        y += 1

    return out

# -------------------------
# Alpha-over compositing (FG over BG) with placement (x,y)
# FG may be RGB (c=3) or RGBA (c=4). BG uses c channels (1 or 3 or 4). Output keeps BG's channel count (if 4, alpha preserved).
# Opacity is a global multiplier in [0..255].
# -------------------------
@staticmethod
fn alpha_over_u8_hwc(bg_h: Int, bg_w: Int, c: Int, BG: List[UInt8],
                      fg_h: Int, fg_w: Int, fg_c: Int, FG: List[UInt8],
                      x0: Int, y0: Int, opacity: Int) -> List[UInt8]:
    if bg_h <= 0 or bg_w <= 0 or c <= 0:
        return List[UInt8]()
    if fg_h <= 0 or fg_w <= 0 or fg_c <= 0:
        # nothing to place; return copy of BG
        var copy: List[UInt8] = List[UInt8]()
        var i = 0; var n = bg_h * bg_w * c
        while i < n: copy.append(BG[i]); i += 1
        return copy

    var out = _alloc_u8(bg_h * bg_w * c)
    # copy BG first
    var i2 = 0; var n2 = bg_h * bg_w * c
    while i2 < n2:
        out[i2] = BG[i2]
        i2 += 1

    var t_global = opacity
    if t_global < 0: t_global = 0
    if t_global > 255: t_global = 255

    var y = 0
    while y < fg_h:
        var by = y0 + y
        if by < 0 or by >= bg_h:
            y += 1
            continue
        var x = 0
        while x < fg_w:
            var bx = x0 + x
            if bx < 0 or bx >= bg_w:
                x += 1
                continue

            # per-pixel alpha from FG if fg_c==4, else 255
            var fg_r = UInt8(0); var fg_g = UInt8(0); var fg_b = UInt8(0); var fg_a = UInt8(255)
            if fg_c == 4:
                var s = _offset(fg_w, 4, x, y, 0)
                fg_r = FG[s + 0]; fg_g = FG[s + 1]; fg_b = FG[s + 2]; fg_a = FG[s + 3]
            elif fg_c == 3:
                var s3 = _offset(fg_w, 3, x, y, 0)
                fg_r = FG[s3 + 0]; fg_g = FG[s3 + 1]; fg_b = FG[s3 + 2]
                fg_a = UInt8(255)
            else:
                # Unsupported FG layout; skip
                x += 1
                continue

            # effective alpha (fixed point 0..255)
            var a_eff = (Int(fg_a) * t_global + 127) // 255

            # BG fetch per channel set
            if c == 1:
                var bg_g = Int(out[_offset(bg_w, 1, bx, by, 0)])
                var fg_gray = (Int(fg_r) * 77 + Int(fg_g) * 150 + Int(fg_b) * 29) >> 8
                var res = ( (255 - a_eff) * bg_g + a_eff * fg_gray + 127 ) // 255
                out[_offset(bg_w, 1, bx, by, 0)] = _clamp_u8(res)
            elif c == 3 or c == 4:
                var bidx = _offset(bg_w, c, bx, by, 0)
                var br = Int(out[bidx + 0])
                var bg = Int(out[bidx + 1])
                var bb = Int(out[bidx + 2])
                var rr = ( (255 - a_eff) * br + a_eff * Int(fg_r) + 127 ) // 255
                var rg = ( (255 - a_eff) * bg + a_eff * Int(fg_g) + 127 ) // 255
                var rb = ( (255 - a_eff) * bb + a_eff * Int(fg_b) + 127 ) // 255
                out[bidx + 0] = _clamp_u8(rr)
                out[bidx + 1] = _clamp_u8(rg)
                out[bidx + 2] = _clamp_u8(rb)
                if c == 4:
                    # keep BG alpha as-is (straight alpha). Optionally could composite alphas: a_out = a_bg + a_fg*(1-a_bg)
                    # Here we leave it unchanged for simplicity.
                    pass
            x += 1
        y += 1

    return out

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # same-size blend: 2x2 RGB
    var h = 2; var w = 2; var c = 3
    var A: List[UInt8] = List[UInt8]()
    var B: List[UInt8] = List[UInt8]()
    # A: red, green, blue, white
    A.append(255); A.append(0);   A.append(0)
    A.append(0);   A.append(255); A.append(0)
    A.append(0);   A.append(0);   A.append(255)
    A.append(255); A.append(255); A.append(255)
    # B: black, black, white, white
    B.append(0);   B.append(0);   B.append(0)
    B.append(0);   B.append(0);   B.append(0)
    B.append(255); B.append(255); B.append(255)
    B.append(255); B.append(255); B.append(255)

    var out_add = blend_same_size_u8_hwc(BlendMode.ADD(), h,w,c, A,B, 128)
    if len(out_add) != 12: return False

    var out_mul = blend_same_size_u8_hwc(BlendMode.MULTIPLY(), h,w,c, A,B, 255)
    if len(out_mul) != 12: return False

    # alpha-over: BG 3x3 gray ramp, FG 2x2 RGBA placed at (1,1)
    var bg_h = 3; var bg_w = 3; var bg_c = 3
    var BG: List[UInt8] = List[UInt8]()
    var i = 0
    while i < bg_h * bg_w:
        var g = UInt8(50 + (i * 10))
        BG.append(g); BG.append(g); BG.append(g)
        i += 1

    var fg_h = 2; var fg_w = 2; var fg_c = 4
    var FG: List[UInt8] = List[UInt8]()
    # RGBA: (255,0,0,128), (0,255,0,128), (0,0,255,128), (255,255,255,128)
    FG.append(255); FG.append(0);   FG.append(0);   FG.append(128)
    FG.append(0);   FG.append(255); FG.append(0);   FG.append(128)
    FG.append(0);   FG.append(0);   FG.append(255); FG.append(128)
    FG.append(255); FG.append(255); FG.append(255); FG.append(128)

    var comp = alpha_over_u8_hwc(bg_h, bg_w, bg_c, BG, fg_h, fg_w, fg_c, FG, 1, 1, 255)
    if len(comp) != bg_h * bg_w * bg_c: return False

    return True