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
# Project: momijo.vision.transforms
# File: src/momijo/vision/transforms/compose.mojo

struct ChannelOrder(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn GRAY() -> ChannelOrder: return ChannelOrder(1)
    @staticmethod fn RGB()  -> ChannelOrder: return ChannelOrder(2)
    @staticmethod fn BGR()  -> ChannelOrder: return ChannelOrder(3)
    @staticmethod fn RGBA() -> ChannelOrder: return ChannelOrder(4)
    @staticmethod fn BGRA() -> ChannelOrder: return ChannelOrder(5)
fn __eq__(self, other: ChannelOrder) -> Bool: return self.id == other.id

@staticmethod
fn _num_ch(order: ChannelOrder) -> Int:
    if order == ChannelOrder.GRAY(): return 1
    if order == ChannelOrder.RGB() or order == ChannelOrder.BGR(): return 3
    return 4

struct Image(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var data: List[UInt8]
fn __init__(out self, h: Int, w: Int, order: ChannelOrder, data: List[UInt8]) -> None:
        self.h = h; self.w = w; self.order = order
        var expected = h * w * _num_ch(order)
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                buf.append(0); i += 1
            self.data = buf
        else:
            self.data = data

# -------------------------
# Enums
# -------------------------
struct Interp(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn NEAREST() -> Interp:  return Interp(0)
    @staticmethod fn BILINEAR() -> Interp: return Interp(1)
fn __eq__(self, other: Interp) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("NEAREST")
        if self.id == 1: return String("BILINEAR")
        return String("Interp(") + String(self.id) + String(")")

struct PadMode(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn ZERO() -> PadMode: return PadMode(0)
    @staticmethod fn EDGE() -> PadMode: return PadMode(1)
fn __eq__(self, other: PadMode) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("ZERO")
        if self.id == 1: return String("EDGE")
        return String("Pad(") + String(self.id) + String(")")

struct TKind(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn Resize()      -> TKind: return TKind(1)
    @staticmethod fn CenterCrop()  -> TKind: return TKind(2)
    @staticmethod fn RGBToGray()   -> TKind: return TKind(3)
    @staticmethod fn BGRToRGB()    -> TKind: return TKind(4)
    @staticmethod fn DropAlpha()   -> TKind: return TKind(5)
    @staticmethod fn ColorJitter() -> TKind: return TKind(6)
fn __eq__(self, other: TKind) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 1: return String("Resize")
        if self.id == 2: return String("CenterCrop")
        if self.id == 3: return String("RGBToGray")
        if self.id == 4: return String("BGRToRGB")
        if self.id == 5: return String("DropAlpha")
        if self.id == 6: return String("ColorJitter")
        return String("TKind(") + String(self.id) + String(")")

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
        out.append(0); i += 1
    return out

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

@staticmethod
fn _clamp_i(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

# -------------------------
# Kernels: convert color
# -------------------------
@staticmethod
fn bgr_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h*w*3)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            out[base+0] = src[base+2]
            out[base+1] = src[base+1]
            out[base+2] = src[base+0]
            x += 1
        y += 1
    return out

@staticmethod
fn rgb_to_gray_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h*w)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            var r = src[base+0]; var g = src[base+1]; var b = src[base+2]
            var gy = UInt16(77)*UInt16(r) + UInt16(150)*UInt16(g) + UInt16(29)*UInt16(b)
            out[y*w + x] = UInt8((gy >> UInt8(8)) & UInt16(0xFF))
            x += 1
        y += 1
    return out

@staticmethod
fn drop_alpha_to_rgb_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8]) -> (ChannelOrder, List[UInt8]):
    # Assumes c in {3,4}. If 4, drop last channel and keep RGB; if 3, copy.
    var out = _alloc_u8(h*w*3)
    if c == 3:
        var i = 0; var n = h*w*3
        while i < n: out[i] = src[i]; i += 1
        return (ChannelOrder.RGB(), out)
    # c==4: copy first 3 channels
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var s = _offset(w, 4, x, y, 0)
            var d = _offset(w, 3, x, y, 0)
            out[d+0] = src[s+0]
            out[d+1] = src[s+1]
            out[d+2] = src[s+2]
            x += 1
        y += 1
    return (ChannelOrder.RGB(), out)

# -------------------------
# Kernels: resize
# -------------------------
@staticmethod
fn resize_nearest_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    var out = _alloc_u8(oh*ow*c)
    var y = 0
    while y < oh:
        var sy = (y * h) // oh
        var x = 0
        while x < ow:
            var sx = (x * w) // ow
            var ch = 0
            while ch < c:
                out[_offset(ow, c, x, y, ch)] = src[_offset(w, c, sx, sy, ch)]
                ch += 1
            x += 1
        y += 1
    return out

@staticmethod
fn resize_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    var out = _alloc_u8(oh*ow*c)
    var y = 0
    while y < oh:
        var sy_fp = 0
        if oh > 1: sy_fp = (y * (h - 1) * 256) // (oh - 1)
        var y0 = sy_fp >> UInt8(8)
        var fy = sy_fp & UInt8(255)
        var y1 = y0 + 1; if y1 >= h: y1 = h - 1
        var wy0 = 256 - fy
        var x = 0
        while x < ow:
            var sx_fp = 0
            if ow > 1: sx_fp = (x * (w - 1) * 256) // (ow - 1)
            var x0 = sx_fp >> UInt8(8)
            var fx = sx_fp & UInt8(255)
            var x1 = x0 + 1; if x1 >= w: x1 = w - 1
            var wx0 = 256 - fx
            var w00 = wy0 * wx0
            var w10 = wy0 * fx
            var w01 = fy * wx0
            var w11 = fy * fx
            var ch = 0
            while ch < c:
                var p00 = Int(src[_offset(w, c, x0, y0, ch)])
                var p10 = Int(src[_offset(w, c, x1, y0, ch)])
                var p01 = Int(src[_offset(w, c, x0, y1, ch)])
                var p11 = Int(src[_offset(w, c, x1, y1, ch)])
                var acc = p00*w00 + p10*w10 + p01*w01 + p11*w11
                var val = (acc + 32768) >> 16
                out[_offset(ow, c, x, y, ch)] = _clamp_u8(val)
                ch += 1
            x += 1
        y += 1
    return out

# -------------------------
# Kernels: center crop
# -------------------------
@staticmethod
fn center_crop_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], out_h: Int, out_w: Int, pad: PadMode) -> List[UInt8]:
    var out = _alloc_u8(out_h*out_w*c)
    var sy0 = (h // 2) - (out_h // 2)
    var sx0 = (w // 2) - (out_w // 2)
    var y = 0
    while y < out_h:
        var x = 0
        while x < out_w:
            var sy = y + sy0
            var sx = x + sx0
            var ch = 0
            if pad == PadMode.EDGE():
                var syc = _clamp_i(sy, 0, h-1)
                var sxc = _clamp_i(sx, 0, w-1)
                while ch < c:
                    out[_offset(out_w, c, x, y, ch)] = src[_offset(w, c, sxc, syc, ch)]
                    ch += 1
            else:
                if sy >= 0 and sy < h and sx >= 0 and sx < w:
                    while ch < c:
                        out[_offset(out_w, c, x, y, ch)] = src[_offset(w, c, sx, sy, ch)]
                        ch += 1
            x += 1
        y += 1
    return out

# -------------------------
# Kernels: color jitter (brightness/contrast/saturation/hue)
# -------------------------
@staticmethod
fn adjust_brightness_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], delta: Int) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w*c:
        # keep alpha if c==4 and (i%4==3)
        if c == 4 and (i % 4) == 3:
            out[i] = src[i]
        else:
            out[i] = _clamp_u8(Int(src[i]) + delta)
        i += 1
    return out

@staticmethod
fn adjust_contrast_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], factor_q8: Int) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w*c:
        if c == 4 and (i % 4) == 3:
            out[i] = src[i]
        else:
            var x = Int(src[i])
            var y = ((x - 128) * factor_q8) >> 8
            out[i] = _clamp_u8(y + 128)
        i += 1
    return out

@staticmethod
fn adjust_saturation_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], factor_q8: Int) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w:
        var r = 0; var g = 0; var b = 0; var a = 255
        if c == 1:
            var v = Int(src[i]); r=v; g=v; b=v
        else:
            r = Int(src[i*c+0]); g = Int(src[i*c+1]); b = Int(src[i*c+2])
            if c == 4: a = Int(src[i*4+3])
        var gray = (77*r + 150*g + 29*b) >> 8
        var rr = gray + (((r - gray) * factor_q8) >> 8)
        var gg = gray + (((g - gray) * factor_q8) >> 8)
        var bb = gray + (((b - gray) * factor_q8) >> 8)
        if c == 1:
            out[i] = _clamp_u8(rr)
        elif c == 3:
            out[i*3+0] = _clamp_u8(rr)
            out[i*3+1] = _clamp_u8(gg)
            out[i*3+2] = _clamp_u8(bb)
        else:
            out[i*4+0] = _clamp_u8(rr)
            out[i*4+1] = _clamp_u8(gg)
            out[i*4+2] = _clamp_u8(bb)
            out[i*4+3] = _clamp_u8(a)
        i += 1
    return out

@staticmethod
fn _deg_to_rad_q16(d: Int) -> Int:
    var pi_q16 = 205887
    return (d * pi_q16) // 180

@staticmethod
fn _wrap_pi_q16(x: Int) -> Int:
    var pi_q16 = 205887
    var tp = pi_q16 * 2
    var r = x % tp
    if r > pi_q16: r = r - tp
    if r < -pi_q16: r = r + tp
    return r

@staticmethod
fn _abs_i(v: Int) -> Int:
    if v < 0: return -v
    return v

@staticmethod
fn _sin_q16(x: Int) -> Int:
    var A = 83443; var B = 26519
    var t = _wrap_pi_q16(x)
    var term1 = (A * t) >> 16
    var term2 = (B * ((t * _abs_i(t)) >> 16)) >> 16
    return term1 - term2

@staticmethod
fn _cos_q16(x: Int) -> Int:
    var p2 = 102943
    return _sin_q16(x + p2)

@staticmethod
fn adjust_hue_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], deg: Int) -> List[UInt8]:
    if deg == 0:
        var copy: List[UInt8] = List[UInt8](); var n = h*w*c; var i0 = 0
        while i0 < n: copy.append(src[i0]); i0 += 1
        return copy
    var ang = _deg_to_rad_q16(deg); var s = _sin_q16(ang); var co = _cos_q16(ang)
    var YR = 4905; var YG = 9629; var YB = 1860
    var IR = 9761; var IG = -4483; var IB = -5272
    var QR = 3452; var QG = -8567; var QB = 5118
    var RY = 15678; var RQ = 10180
    var GY = 16384; var GI = -4461; var GQ = -10603
    var BY = 16384; var BI = -18125; var BQ = 27904

    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w:
        var r = 0; var g = 0; var b = 0; var a = 255
        if c == 1:
            var v = Int(src[i]); r=v; g=v; b=v
        else:
            r = Int(src[i*c+0]); g = Int(src[i*c+1]); b = Int(src[i*c+2])
            if c == 4: a = Int(src[i*c+3])
        var Y = YR*r + YG*g + YB*b
        var I = IR*r + IG*g + IB*b
        var Q = QR*r + QG*g + QB*b
        var I2 = ((I * co) >> 16) - ((Q * s) >> 16)
        var Q2 = ((I * s) >> 16) + ((Q * co) >> 16)
        var r14 = Y + (RY * I2 >> UInt8(14)) + (RQ * Q2 >> UInt8(14))
        var g14 = (GY * (Y >> UInt8(14)) << 14) + (GI * I2 >> UInt8(14)) + (GQ * Q2 >> UInt8(14))
        var b14 = (BY * (Y >> UInt8(14)) << 14) + (BI * I2 >> UInt8(14)) + (BQ * Q2 >> UInt8(14))
        var rr = r14 >> UInt8(14); var gg = g14 >> UInt8(14); var bb = b14 >> UInt8(14)
        if c == 1:
            var gy = (77*rr + 150*gg + 29*bb) >> 8
            out[i] = _clamp_u8(gy)
        elif c == 3:
            out[i*3+0] = _clamp_u8(rr)
            out[i*3+1] = _clamp_u8(gg)
            out[i*3+2] = _clamp_u8(bb)
        else:
            out[i*4+0] = _clamp_u8(rr)
            out[i*4+1] = _clamp_u8(gg)
            out[i*4+2] = _clamp_u8(bb)
            out[i*4+3] = _clamp_u8(a)
        i += 1
    return out

@staticmethod
fn apply_color_jitter_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8],
                             bd: Int, cq8: Int, sq8: Int, hd: Int) -> List[UInt8]:
    var buf = src
    if bd != 0: buf = adjust_brightness_u8_hwc(h,w,c,buf,bd)
    if cq8 != 256: buf = adjust_contrast_u8_hwc(h,w,c,buf,cq8)
    if sq8 != 256: buf = adjust_saturation_u8_hwc(h,w,c,buf,sq8)
    if hd != 0: buf = adjust_hue_u8_hwc(h,w,c,buf,hd)
    return buf

# -------------------------
# Transform & Pipeline
# -------------------------
struct Transform(Copyable, Movable):
    var kind: TKind
    var p1: Int
    var p2: Int
    var p3: Int
    var p4: Int
fn __init__(out self, kind: TKind) -> None:
        self.kind = kind; self.p1 = 0; self.p2 = 0; self.p3 = 0; self.p4 = 0
fn with_i(mut self, idx: Int, v: Int) -> Transform:
        if idx == 1: self.p1 = v
        elif idx == 2: self.p2 = v
        elif idx == 3: self.p3 = v
        else: self.p4 = v
        return self
fn to_string(self) -> String:
        return String("Transform(") + self.kind.to_string() + String(", ") + String(self.p1) + String(",") + String(self.p2) + String(",") + String(self.p3) + String(",") + String(self.p4) + String(")")

struct Pipeline(Copyable, Movable):
    var ops: List[Transform]
fn __init__(out self) -> None:
        self.ops = List[Transform]()
fn add(mut self, t: Transform) -> Pipeline:
        self.ops.append(t); return self
fn summary(self) -> String:
        var s = String("Pipeline[") + String(len(self.ops)) + String("]\n")
        var i = 0
        while i < len(self.ops):
            s = s + String("  ") + self.ops[i].to_string() + String("\n")
            i += 1
        return s
fn run(self, img: Image) -> (Bool, Image):
        var cur = img
        var i = 0
        while i < len(self.ops):
            var t = self.ops[i]
            var ok = False; var out_img = Image(0,0, img.order, List[UInt8]())
            (ok, out_img) = _apply_one(t, cur)
            if not ok:
                return (False, cur)
            cur = out_img
            i += 1
        return (True, cur)

# -------------------------
# Application of a single transform
# -------------------------
@staticmethod
fn _apply_one(t: Transform, img: Image) -> (Bool, Image):
    var kind = t.kind
    if kind == TKind.Resize():
        var out_h = t.p1; var out_w = t.p2; var mode = t.p3
        if out_h <= 0 or out_w <= 0: return (False, img)
        var c = _num_ch(img.order)
        var buf: List[UInt8] = List[UInt8]()
        if mode == 0:
            buf = resize_nearest_u8_hwc(img.h, img.w, c, img.data, out_h, out_w)
        else:
            buf = resize_bilinear_u8_hwc(img.h, img.w, c, img.data, out_h, out_w)
        return (True, Image(out_h, out_w, img.order, buf))

    if kind == TKind.CenterCrop():
        var out_h = t.p1; var out_w = t.p2; var pad = t.p3
        if out_h <= 0 or out_w <= 0: return (False, img)
        var c = _num_ch(img.order)
        var padm = PadMode.ZERO()
        if pad != 0: padm = PadMode.EDGE()
        var buf = center_crop_u8_hwc(img.h, img.w, c, img.data, out_h, out_w, padm)
        return (True, Image(out_h, out_w, img.order, buf))

    if kind == TKind.RGBToGray():
        # If not RGB, try to coerce: BGR -> RGB first; RGBA/BGRA -> drop alpha then assume RGB
        var ord = img.order
        if ord == ChannelOrder.GRAY():
            var copy: List[UInt8] = List[UInt8](); var n = img.h*img.w; var i0 = 0
            while i0 < n: copy.append(img.data[i0]); i0 += 1
            return (True, Image(img.h, img.w, ChannelOrder.GRAY(), copy))
        var rgb: List[UInt8] = List[UInt8]()
        if ord == ChannelOrder.RGB():
            var n3 = img.h*img.w*3; var i1 = 0
            while i1 < n3: rgb.append(img.data[i1]); i1 += 1
        elif ord == ChannelOrder.BGR():
            rgb = bgr_to_rgb_u8_hwc(img.h, img.w, img.data)
        else:
            var c = _num_ch(ord)
            var tmp: List[UInt8] = List[UInt8]()
            var y = 0
            while y < img.h:
                var x = 0
                while x < img.w:
                    var base = _offset(img.w, c, x, y, 0)
                    tmp.append(img.data[base+0]); tmp.append(img.data[base+1]); tmp.append(img.data[base+2])
                    x += 1
                y += 1
            rgb = tmp
        var gbuf = rgb_to_gray_u8_hwc(img.h, img.w, rgb)
        return (True, Image(img.h, img.w, ChannelOrder.GRAY(), gbuf))

    if kind == TKind.BGRToRGB():
        if img.order == ChannelOrder.RGB():
            var copy: List[UInt8] = List[UInt8](); var n = img.h*img.w*3; var i2 = 0
            while i2 < n: copy.append(img.data[i2]); i2 += 1
            return (True, Image(img.h, img.w, ChannelOrder.RGB(), copy))
        if img.order != ChannelOrder.BGR():
            return (False, img)
        var buf = bgr_to_rgb_u8_hwc(img.h, img.w, img.data)
        return (True, Image(img.h, img.w, ChannelOrder.RGB(), buf))

    if kind == TKind.DropAlpha():
        var c = _num_ch(img.order)
        if c == 3:
            var copy3: List[UInt8] = List[UInt8](); var n3 = img.h*img.w*3; var ii = 0
            while ii < n3: copy3.append(img.data[ii]); ii += 1
            return (True, Image(img.h, img.w, img.order, copy3))
        if c != 4: return (False, img)
        var new_order: ChannelOrder = ChannelOrder.RGB()
        var buf3: List[UInt8] = List[UInt8]()
        (new_order, buf3) = drop_alpha_to_rgb_u8_hwc(img.h, img.w, c, img.data)
        return (True, Image(img.h, img.w, new_order, buf3))

    if kind == TKind.ColorJitter():
        # p1=brightness_delta, p2=contrast_q8, p3=saturation_q8, p4=hue_deg
        var bd = t.p1; var cq8 = t.p2; var sq8 = t.p3; var hd = t.p4
        var c = _num_ch(img.order)
        var buf = apply_color_jitter_u8_hwc(img.h, img.w, c, img.data, bd, cq8, sq8, hd)
        return (True, Image(img.h, img.w, img.order, buf))

    return (False, img)

# -------------------------
# Builders for transforms
# -------------------------
@staticmethod
fn T_resize(out_h: Int, out_w: Int, mode: Interp) -> Transform:
    var t = Transform(TKind.Resize())
    t = t.with_i(1, out_h).with_i(2, out_w).with_i(3, mode.id)
    return t

@staticmethod
fn T_center_crop(out_h: Int, out_w: Int, pad: PadMode) -> Transform:
    var t = Transform(TKind.CenterCrop())
    t = t.with_i(1, out_h).with_i(2, out_w).with_i(3, pad.id)
    return t

@staticmethod
fn T_rgb_to_gray() -> Transform:
    return Transform(TKind.RGBToGray())

@staticmethod
fn T_bgr_to_rgb() -> Transform:
    return Transform(TKind.BGRToRGB())

@staticmethod
fn T_drop_alpha() -> Transform:
    return Transform(TKind.DropAlpha())

@staticmethod
fn T_color_jitter(bd: Int, cq8: Int, sq8: Int, hd: Int) -> Transform:
    var t = Transform(TKind.ColorJitter())
    t = t.with_i(1, bd).with_i(2, cq8).with_i(3, sq8).with_i(4, hd)
    return t

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 2x2 BGR with RGBA alpha pattern to exercise ops
    var bgr: List[UInt8] = List[UInt8]()
    # (B,G,R): (0,0,255) (0,255,0) (255,0,0) (255,255,255)
    bgr.append(0); bgr.append(0);   bgr.append(255)
    bgr.append(0); bgr.append(255); bgr.append(0)
    bgr.append(255); bgr.append(0); bgr.append(0)
    bgr.append(255); bgr.append(255); bgr.append(255)
    var img_bgr = Image(2, 2, ChannelOrder.BGR(), bgr)

    var pipe = Pipeline()
    pipe = pipe.add(T_bgr_to_rgb())
    pipe = pipe.add(T_center_crop(2, 2, PadMode.ZERO()))
    pipe = pipe.add(T_resize(3, 3, Interp.BILINEAR()))
    pipe = pipe.add(T_color_jitter(10, 256, 256, 0))
    var ok = False; var out_img = Image(0,0, ChannelOrder.RGB(), List[UInt8]())
    (ok, out_img) = pipe.run(img_bgr)
    if not ok: return False
    if not (out_img.h == 3 and out_img.w == 3 and _num_ch(out_img.order) == 3): return False

    var pipe2 = Pipeline()
    pipe2 = pipe2.add(T_bgr_to_rgb()).add(T_rgb_to_gray())
    var ok2 = False; var out2 = Image(0,0, ChannelOrder.GRAY(), List[UInt8]())
    (ok2, out2) = pipe2.run(img_bgr)
    if not ok2: return False
    if not (_num_ch(out2.order) == 1 and len(out2.data) == 4): return False

    # DropAlpha path
    var rgba: List[UInt8] = List[UInt8]()
    # px0..px3 RGBA
    rgba.append(1); rgba.append(2); rgba.append(3); rgba.append(4)
    rgba.append(5); rgba.append(6); rgba.append(7); rgba.append(8)
    rgba.append(9); rgba.append(10); rgba.append(11); rgba.append(12)
    rgba.append(13); rgba.append(14); rgba.append(15); rgba.append(16)
    var img_rgba = Image(2,2, ChannelOrder.RGBA(), rgba)
    var pipe3 = Pipeline().add(T_drop_alpha()).add(T_resize(1,4, Interp.NEAREST()))
    var ok3 = False; var out3 = Image(0,0, ChannelOrder.RGB(), List[UInt8]())
    (ok3, out3) = pipe3.run(img_rgba)
    if not ok3: return False
    if not (out3.h == 1 and out3.w == 4 and _num_ch(out3.order) == 3): return False

    return True