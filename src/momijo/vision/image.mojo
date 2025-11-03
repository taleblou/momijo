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
# File: src/momijo/vision/image.mojo

from momijo.vision.dtypes import DType, Layout, ColorSpace
from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.memory import alloc_u8

# -----------------------------------------------------------------------------
# Image metadata
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ImageMeta: layout + colorspace + alpha-premultiplication flag
# -----------------------------------------------------------------------------
struct ImageMeta(Copyable, Movable):
    var _layout: Layout
    var _cs: ColorSpace
    var _alpha_premultiplied: Bool

    # Defaultable constructor (enables ImageMeta())
    fn __init__(
        out self,
        layout: Layout = Layout.HWC(),
        cs: ColorSpace = ColorSpace.SRGB(),
        alpha_premultiplied: Bool = False
    ):
        # Defensive copies to avoid aliasing
        self._layout = layout.copy()
        self._cs = cs.copy()
        self._alpha_premultiplied = alpha_premultiplied

    fn __copyinit__(out self, other: Self):
        self._layout = other._layout.copy()
        self._cs = other._cs.copy()
        self._alpha_premultiplied = other._alpha_premultiplied

    # Explicit copy helper (keeps style parity with Image.copy())
    fn copy(self) -> ImageMeta:
        return ImageMeta(self._layout.copy(), self._cs.copy(), self._alpha_premultiplied)

    # --- getters ---
    fn layout(self) -> Layout:
        return self._layout.copy()

    fn colorspace(self) -> ColorSpace:
        return self._cs.copy()

    fn is_premultiplied(self) -> Bool:
        return self._alpha_premultiplied

    # Optional synonym for readability in some call sites
    fn alpha_premultiplied(self) -> Bool:
        return self._alpha_premultiplied

    # --- immutable-style transformers (return new ImageMeta) ---
    fn with_colorspace(self, cs: ColorSpace) -> ImageMeta:
        return ImageMeta(self._layout.copy(), cs.copy(), self._alpha_premultiplied)

    fn with_layout(self, layout: Layout) -> ImageMeta:
        return ImageMeta(layout.copy(), self._cs.copy(), self._alpha_premultiplied)

    fn with_premultiplied(self, v: Bool) -> ImageMeta:
        return ImageMeta(self._layout.copy(), self._cs.copy(), v)

    # --- mutating setters (when in-place update is desirable) ---
    fn set_colorspace(mut self, cs: ColorSpace):
        self._cs = cs.copy()

    fn set_layout(mut self, layout: Layout):
        self._layout = layout.copy()

    fn set_premultiplied(mut self, v: Bool):
        self._alpha_premultiplied = v

    # Human-readable string summary
    fn __str__(self) -> String:
        var layout_s = self._layout.__str__()
        var cs_s = self._cs.__str__()
        var premul = String("False")
        if self._alpha_premultiplied:
            premul = String("True")

        var out = String("ImageMeta(")
        out += String("layout=") + layout_s
        out += String(", colorspace=") + cs_s
        out += String(", premultiplied=") + premul
        out += String(")")
        return out


# -----------------------------------------------------------------------------
# Image wrapper around a Tensor
# -----------------------------------------------------------------------------
struct Image(Copyable, Movable):
    var _tensor: Tensor
    var _meta: ImageMeta

    fn __init__(out self, meta: ImageMeta, tensor: Tensor):
        # Defensive copies to avoid shared ownership pitfalls
        self._meta = meta.copy()
        self._tensor = tensor.clone_deep()

    fn __copyinit__(out self, other: Self):
        self._tensor = other._tensor.clone_deep()
        self._meta = other._meta.copy()


    # Human-readable summary
    fn __str__(self) -> String:
        var h = self.height()
        var w = self.width()
        var c = self.channels()

        var layout_s = self._meta.layout().__str__()        # Layout(...)
        var cs_s     = self._meta.colorspace().__str__()    # ColorSpace(...)
        var dt_s     = self._tensor.dtype().__str__()       # DType(...)

        var s0 = self._tensor.stride0()
        var s1 = self._tensor.stride1()
        var s2 = self._tensor.stride2()

        var packed_s = String("False")
        if self.is_contiguous_hwc_u8():
            packed_s = String("True")

        var out = String("Image(")
        out += String(h) + String("x") + String(w) + String("x") + String(c)
        out += String(", layout=") + layout_s
        out += String(", colorspace=") + cs_s
        out += String(", dtype=") + dt_s
        out += String(", strides=(") + String(s0) + String(",") + String(s1) + String(",") + String(s2) + String(")")
        out += String(", packed_hwc_u8=") + packed_s
        out += String(")")
        return out

    # --- basic getters ---
    fn tensor(self) -> Tensor:
        return self._tensor.copy()

    fn meta(self) -> ImageMeta:
        return self._meta.copy()

    fn height(self) -> Int:
        return self._tensor.height()

    fn width(self) -> Int:
        return self._tensor.width()

    fn channels(self) -> Int:
        return self._tensor.channels()

    fn dtype(self) -> DType:
        return self._tensor.dtype()

    fn colorspace(self) -> ColorSpace:
        return self._meta.colorspace()

    fn layout(self) -> Layout:
        return self._meta.layout()

    # --- layout / contiguity helpers ---
    fn is_empty(self) -> Bool:
        return self.height() <= 0 or self.width() <= 0 or self.channels() <= 0

    fn is_hwc(self) -> Bool:
        return self._meta.layout() == Layout.HWC()

    fn is_u8(self) -> Bool:
        return self._tensor.dtype() == DType.UInt8()

    fn is_contiguous_hwc_u8(self) -> Bool:
        return self.is_hwc() and self.is_u8() and self._tensor.is_contiguous_hwc_u8()

    # if packed HWC/UInt8; clone if already packed and copy_if_needed=True.
    # Image -> HWC UInt8 packed (copy-based, no channel swap)
    # Always returns a valid HWC-UInt8 image with strides=(W*C, C, 1) in RGB order.
    # Packed HWC UInt8 سازگار و ایمن
    fn ensure_packed_hwc_u8(self, force: Bool) -> Image:
        var H = self.height()
        var W = self.width()
        var C = self.channels()
        # ابعاد نامعتبر => همان کپی سبک
        if H <= 0 or W <= 0 or C <= 0:
            return self.copy()

        # فرض: اگر layout==HWC و dtype==UInt8 و stride2==1، stride1==C، stride0==W*C

        # تخصیص مقصد: قبل از ساخت، ضرب ایمن
        var ok_sz = _safe_mul3(H, W, C)
        if not ok_sz[0]:
            return self.copy()

        var out = Image.new_hwc_u8(H, W, C, UInt8(0), self.colorspace(), Layout.HWC())

        var y = 0
        while y < H:
            var x = 0
            while x < W:
                var k = 0
                while k < C:
                    out.set_u8(y, x, k, self.get_u8(y, x, k))  # RGB: 0=R,1=G,2=B
                    k += 1
                x += 1
            y += 1

        return out.copy()









    # --- cloning / views ---
    fn clone(self) -> Image:
        var copied = self._tensor.clone()
        return Image(self._meta.copy(), copied.copy())

    fn roi(self, y: Int, x: Int, h: Int, w: Int) -> Image:
        # Safe ROI for HWC; clamps to bounds. Fallback to copy when not HWC.
        if not self.is_hwc():
            return self.copy()

        var H = self.height()
        var W = self.width()
        if H <= 0 or W <= 0:
            return self.copy()

        var y0 = y
        if y0 < 0: y0 = 0
        if y0 >= H: y0 = H - 1

        var x0 = x
        if x0 < 0: x0 = 0
        if x0 >= W: x0 = W - 1

        var y1 = y + h
        if y1 <= y0 + 1: y1 = y0 + 1
        if y1 > H: y1 = H

        var x1 = x + w
        if x1 <= x0 + 1: x1 = x0 + 1
        if x1 > W: x1 = W

        var hh = y1 - y0
        var ww = x1 - x0

        var v = self._tensor.copy_roi(y0, x0, hh, ww)
        return Image(self._meta.copy(), v.copy())

    # Keep API parity; currently returns a deep copy (no zero-copy view yet).
    fn as_hwc_view(self, h: Int, w: Int, c: Int, s0: Int, s1: Int, s2: Int) -> Image:
        return self.copy()

    # --- meta transforms ---
    fn with_meta(self, meta: ImageMeta) -> Image:
        return Image(meta.copy(), self._tensor.copy())

    fn with_colorspace(self, cs: ColorSpace) -> Image:
        var m = self._meta.with_colorspace(cs)
        return Image(m.copy(), self._tensor.copy())

    # --- pixel access (requires packed HWC/UInt8); guarded and effective. ---
    fn set_u8(self, y: Int, x: Int, k: Int, v: UInt8):
        var t = self._tensor.copy()
        var off = y * t.stride0() + x * t.stride1() + k * t.stride2()
        var p = t.data()                 # UnsafePointer[UInt8]
        p[off] = v

    fn at_u8(self, y: Int, x: Int, ch: Int) -> UInt8:
        if not self.is_contiguous_hwc_u8():
            return UInt8(0)

        var H = self.height()
        var W = self.width()
        var C = self.channels()
        if y < 0 or y >= H: return UInt8(0)
        if x < 0 or x >= W: return UInt8(0)
        if ch < 0 or ch >= C: return UInt8(0)

        var idx = (y * W + x) * C + ch
        return self._tensor.load_u8_at(idx)

    fn get_u8(self, y: Int, x: Int, k: Int) -> UInt8:
        var t = self._tensor.copy()
        var off = y * t.stride0() + x * t.stride1() + k * t.stride2()
        var p = t.data()                 # UnsafePointer[UInt8]
        return p[off]

    fn set_rgb_u8(mut self, y: Int, x: Int, r: UInt8, g: UInt8, b: UInt8):
        self.set_u8(y, x, 0, r)
        self.set_u8(y, x, 1, g)
        self.set_u8(y, x, 2, b)

    fn get_rgb_u8(self, y: Int, x: Int) -> (UInt8, UInt8, UInt8):
        var r = self.get_u8(y, x, 0)
        var g = self.get_u8(y, x, 1)
        var b = self.get_u8(y, x, 2)
        return (r, g, b)

    # --- alpha utilities (only when channels == 4, packed HWC/UInt8) ---
    fn premultiply_alpha(self) -> Image:
        if not self.is_contiguous_hwc_u8(): return self.copy()
        if self.channels() != 4: return self.copy()

        var out = self.clone()
        var h = out.height()
        var w = out.width()
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var a = out.get_u8(y, x, 3)
                var r = out.get_u8(y, x, 0)
                var g = out.get_u8(y, x, 1)
                var b = out.get_u8(y, x, 2)

                var rr = (Int(r) * Int(a) + 127) // 255
                var gg = (Int(g) * Int(a) + 127) // 255
                var bb = (Int(b) * Int(a) + 127) // 255

                if rr > 255: rr = 255
                if gg > 255: gg = 255
                if bb > 255: bb = 255

                out.set_u8(y, x, 0, UInt8(rr))
                out.set_u8(y, x, 1, UInt8(gg))
                out.set_u8(y, x, 2, UInt8(bb))
                x += 1
            y += 1
        return out.copy()

    fn unpremultiply_alpha(self) -> Image:
        if not self.is_contiguous_hwc_u8(): return self.copy()
        if self.channels() != 4: return self.copy()

        var out = self.clone()
        var h = out.height()
        var w = out.width()
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var a = out.get_u8(y, x, 3)
                if a != UInt8(0):
                    var r = out.get_u8(y, x, 0)
                    var g = out.get_u8(y, x, 1)
                    var b = out.get_u8(y, x, 2)

                    var rr = (Int(r) * 255 + Int(a) // 2) // Int(a)
                    var gg = (Int(g) * 255 + Int(a) // 2) // Int(a)
                    var bb = (Int(b) * 255 + Int(a) // 2) // Int(a)

                    if rr > 255: rr = 255
                    if gg > 255: gg = 255
                    if bb > 255: bb = 255

                    out.set_u8(y, x, 0, UInt8(rr))
                    out.set_u8(y, x, 1, UInt8(gg))
                    out.set_u8(y, x, 2, UInt8(bb))
                x += 1
            y += 1
        return out.copy()


    @always_inline
    fn _flat_index_hwc_u8(self, y: Int, x: Int, ch: Int) -> Int:
        return (y * self.width() + x) * self.channels() + ch

    fn unsafe_set_u8(mut self, y: Int, x: Int, ch: Int, v: UInt8) -> None:
        # Even the "unsafe" variant writes via Tensor's mutating API to avoid
        # pointer-to-temporary pitfalls.
        var idx = self._flat_index_hwc_u8(y, x, ch)
        self._tensor.store_u8_at(idx, v)

    fn unsafe_get_u8(self, y: Int, x: Int, ch: Int) -> UInt8:
        var idx = self._flat_index_hwc_u8(y, x, ch)
        return self._tensor.load_u8_at(idx)


    # Pretty-print full image data as a table of pixel tokens per row.
    # max_rows/max_cols <= 0 means "no limit".
    fn dump_table(self, max_rows: Int = 0, max_cols: Int = 0):
        # if we are in HWC/UInt8 contiguous to simplify pointer math
        var img = self.ensure_packed_hwc_u8(False)

        var H = img.height()
        var W = img.width()
        var C = img.channels()

        if H <= 0 or W <= 0 or C <= 0:
            print("Image is empty.")
            return

        # Pointer to underlying UInt8 buffer
        var p = img._tensor.data()

        # Determine print limits
        var R = H
        var CC = W
        if max_rows > 0 and max_rows < R: R = max_rows
        if max_cols > 0 and max_cols < CC: CC = max_cols

        # Print a short header about the image
        var hdr = String("=== Image dump === ")
        hdr += String(H) + String("x") + String(W) + String("x") + String(C)
        hdr += String(" (HWC/UInt8)")
        if (R != H) or (CC != W):
            hdr += String(" [showing ") + String(R) + String(" rows, ") + String(CC) + String(" cols]")
        print(hdr)

        # Column header
        print(_make_col_header(W, CC))

        # Per-row printing
        var y = 0
        while y < R:
            # Row prefix with y index
            var line = String("y=")
            # pad y to 4 digits
            if y < 10:       line += String("000")
            elif y < 100:    line += String("00")
            elif y < 1000:   line += String("0")
            line += String(y) + String(": ")

            var x = 0
            while x < CC:
                var base = (y * W + x) * C
                # Specialized fast paths for common channel counts
                if C == 1:
                    # Gray: "[ggg]"
                    line += String("[") + _pad3(Int(p[base])) + String("]")
                elif C == 3:
                    # RGB: "[rrr,ggg,bbb]"
                    line += String("[") +
                            _pad3(Int(p[base + 0])) + String(",") +
                            _pad3(Int(p[base + 1])) + String(",") +
                            _pad3(Int(p[base + 2])) + String("]")
                elif C == 4:
                    # RGBA: "[rrr,ggg,bbb,aaa]"
                    line += String("[") +
                            _pad3(Int(p[base + 0])) + String(",") +
                            _pad3(Int(p[base + 1])) + String(",") +
                            _pad3(Int(p[base + 2])) + String(",") +
                            _pad3(Int(p[base + 3])) + String("]")
                else:
                    # Generic C: "[c0,c1,...,c{C-1}]"
                    line += _fmt_pixel_generic_u8(p, base, C)

                # Pixel spacing: one space between tokens
                if x + 1 < CC:
                    line += String(" ")
                x += 1

            print(line)
            y += 1

    @staticmethod
    fn new_hwc_u8(h: Int,w: Int,c: Int,value: UInt8 = UInt8(0),cs: ColorSpace = ColorSpace.SRGB(),layout: Layout = Layout.HWC()) -> Image:
        return new_hwc_u8(h,w,c,value,cs,layout)

    # =============== Free helpers (place in the same module, outside the struct) ===============

@always_inline
fn _safe_mul3(a: Int, b: Int, c: Int) -> (Bool, Int):
    if a <= 0 or b <= 0 or c <= 0: return (False, 0)
    # سقف محافظه‌کارانه: 20000x20000x4 ≈ 1.6e9 بایت
    var MAX = 1_600_000_000
    # ضرب مرحله‌ای با چک
    var ab = a * b
    if ab <= 0 or ab > MAX: return (False, 0)
    var abc = ab * c
    if abc <= 0 or abc > MAX: return (False, 0)
    return (True, abc)


@always_inline
fn _is_packed_hwc_u8(img: Image) -> Bool:
    # layout must be HWC, dtype=UInt8, and strides=(W*C, C, 1)
    if img.layout().id != Layout.HWC().id:
        return False
    if img.dtype().id != DType.UInt8().id:
        return False

    var H = img.height()
    var W = img.width()
    var C = img.channels()
    if H <= 0 or W <= 0 or C <= 0:
        return False

    var s0 = img._tensor._shape0
    var s1 = img._tensor._shape1
    var s2 = img._tensor._shape2
    return (s0 == W * C) and (s1 == C) and (s2 == 1)



fn new_hwc_u8(h: Int, w: Int, c: Int,
              value: UInt8 = UInt8(0),
              cs: ColorSpace = ColorSpace.SRGB(),
              layout: Layout = Layout.HWC()) -> Image:
    # 1) یک Image «خالی» با متا بساز (تنسورش را پروژه‌ات هر طور می‌سازد)
    var m = ImageMeta(layout, cs, False)
    var t = _alloc_tensor_u8(h, w, c, value)
    var img = Image(m.copy(), t.copy())

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var k = 0
            while k < c:
                img.set_u8(y, x, k, value)
                k += 1
            x += 1
        y += 1

    return img.copy()#.ensure_packed_hwc_u8(True)



fn full_hwc_u8(h: Int, w: Int, c: Int, value: UInt8) -> Image:
    return Image.new_hwc_u8(h, w, c, value)


fn zeros_hwc_u8(h: Int, w: Int, c: Int) -> Image:
    return Image.new_hwc_u8(h, w, c, UInt8(0))


@always_inline
fn _pad3(n: Int) -> String:
    # Zero-padded 3-digit decimal for compact alignment (0..255 => "000".."255")
    var s = String(n)
    if n < 10:  return String("00") + s
    if n < 100: return String("0")  + s
    return s

fn _fmt_pixel_generic_u8(ptr: UnsafePointer[UInt8], base: Int, c: Int) -> String:
    var out = String("[")
    var k = 0
    while k < c:
        if k > 0: out += String(",")
        out += _pad3(Int(ptr[base + k]))
        k += 1
    out += String("]")
    return out

# Optional compact header maker for columns
fn _make_col_header(w: Int, max_cols: Int) -> String:
    var W = w
    var MC = max_cols
    if MC > 0 and MC < W:
        W = MC
    var out = String("      x=")  # left margin aligns with row prefix "y=XXXX:"
    var x = 0
    while x < W:
        # 5 chars per pixel token min: "[000]" or "[000,..." etc; we just print index roughly aligned
        # Use 6 spaces per pixel slot for readability.
        var idx = String(x)
        # Left-pad to width 6
        var slot = String("")
        var pad = 6 - len(idx)
        var i = 0
        while i < pad:
            slot += String(" ")
            i += 1
        slot += idx
        out += slot
        x += 1
    return out

# -----------------------------------------------------------------------------
# Helpers / factory functions
# -----------------------------------------------------------------------------

# Allocate a Tensor that is packed HWC/UInt8 and optionally filled with 'value'
fn _alloc_tensor_u8(h: Int, w: Int, c: Int, value: UInt8) -> Tensor:
    var hh = h
    if hh <= 0:
        hh = 1

    var ww = w
    if ww <= 0:
        ww = 1

    var cc = c
    if cc <= 0:
        cc = 1

    var (s0, s1, s2) = packed_hwc_strides(hh, ww, cc)
    var n = hh * ww * cc
    var buf = alloc_u8(n)

    var i = 0
    while i < n:
        buf[i] = value
        i += 1

    return Tensor(buf, n, hh, ww, cc, s0, s1, s2, DType.UInt8())


# Wrap an existing UInt8 buffer in packed HWC
fn make_u8_hwc(out img: Image, data: UnsafePointer[UInt8], h: Int, w: Int, c: Int, cs: ColorSpace):
    if h <= 0 or w <= 0 or c <= 0:
        # Produce an empty image if shape is invalid
        var empty_t = Tensor(UnsafePointer[UInt8](), 0, 0, 0, 0, 0, 0, 0, DType.UInt8(), False)
        var empty_m = ImageMeta(Layout.HWC(), cs, False)
        img = Image(empty_m.copy(),empty_t.copy())
        return

    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var byte_len = h * w * c
    # owns = False because 'data' is external
    var t = Tensor(data, byte_len, h, w, c, s0, s1, s2, DType.UInt8(), False)
    var m = ImageMeta(Layout.HWC(), cs, False)
    img = Image(m,t)

# Allocate a fresh zero-initialized u8 HWC buffer and wrap as Image
# Create a zero-initialized HWC UInt8 image.
fn make_zero_u8_hwc(h: Int, w: Int, c: Int, cs: ColorSpace) -> Image:
    # Empty image fast-path
    if h <= 0 or w <= 0 or c <= 0:
        var empty_t = Tensor(
            UnsafePointer[UInt8](),  # data
            0,                       # length
            0,                       # H
            0,                       # W
            0,                       # C
            0, 0, 0,                 # strides
            DType.UInt8(),           # dtype
            False                    # owns_data
        )
        var empty_m = ImageMeta(Layout.HWC(), cs, False)
        return Image(empty_m.copy(),empty_t.copy())

    # Compute packed strides and allocate buffer
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c
    var buf = alloc_u8(n)

    # Zero-initialize
    var i = 0
    while i < n:
        buf[i] = UInt8(0)
        i += 1

    # Build tensor + metadata and return Image
    var t = Tensor(
        buf,              # data
        n,                # length
        h,                # H
        w,                # W
        c,                # C
        s0, s1, s2,       # strides
        DType.UInt8(),    # dtype
        True              # owns_data
    )
    var m = ImageMeta(Layout.HWC(), cs, False)
    return Image(m.copy(), t.copy())


# Validate basic invariants; returns True if OK (no assertions)
fn validate_image(self: Image) -> Bool:
    if self.height() <= 0: return False
    if self.width()  <= 0: return False
    if self.channels() <= 0: return False
    if not (self.layout() == Layout.HWC()): return False
    if not (self.dtype() == DType.UInt8()): return False
    return True
