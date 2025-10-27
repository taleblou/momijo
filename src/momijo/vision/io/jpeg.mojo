# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/jpeg.mojo
# Description: JPEG read/write using pure Mojo encoder/decoder (file_io backend)

from momijo.vision.tensor import Tensor
from momijo.vision.image import Image, ImageMeta, ColorSpace
from momijo.vision.transforms.array import full
from momijo.vision.io.decode_jpeg import decode_jpeg
from momijo.vision.io.encode_jpeg import encode_jpeg
from momijo.vision.io.file_io import _read_file_bytes, _write_file_raw,_basename,_trim_spaces

from momijo.vision.io.dtypes import DataDType
# ---------------------------- Private helpers ----------------------------

@always_inline
fn _clamp_quality(q: Int) -> Int:
    var v = q
    if v < 1:
        v = 1
    if v > 100:
        v = 100
    return v

@always_inline
fn _to_lower(s: String) -> String:
    var out = String()
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch >= 'A' and ch <= 'Z':
            ch = Char(Int(ch) + (Int('a') - Int('A')))
        out = out + String(ch)
        i += 1
    return out.copy()

@always_inline
fn _ends_with_lower(s: String, suffix: String) -> Bool:
    var a = _to_lower(s)
    var b = _to_lower(suffix)
    var n = a.__len__()
    var m = b.__len__()
    if m > n:
        return False
    var i = 0
    while i < m:
        if a[n - m + i] != b[i]:
            return False
        i += 1
    return True

@always_inline
fn _make_dummy_u8_hwc(h: Int, w: Int, c: Int) -> Tensor:
    # Packed HWC UInt8 tensor filled with mid-gray
    return full((h, w, c), UInt8(127)).tensor()

# ---------------------------- Public API: Capabilities --------------------

fn has_jpeg_codec() -> Bool:
    return True

# ---------------------------- Public API: Read ----------------------------

fn read_jpeg(path: String) -> (Bool, Tensor):
    # _read_file_bytes returns (ok, bytes: List[UInt8])
    var req = _read_file_bytes(path)
    var ok_read = req[0]
    var bytes = req[1].copy()
    if not ok_read:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    var n = len(bytes)
    if n <= 0:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    var ptr = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        ptr[i] = bytes[i]
        i += 1

    var reqd = decode_jpeg(ptr, n)
    var ok_dec = reqd[0]
    var img = reqd[1].copy()

    # free input buffer
    UnsafePointer[UInt8].free(ptr)

    if not ok_dec:
        return (False, _make_dummy_u8_hwc(32, 32, 1))

    # decode_jpeg returns Image; expose its tensor
    return (True, img.tensor())

fn read_jpeg_with_fallback(path: String,
                           fallback_h: Int = 64,
                           fallback_w: Int = 64,
                           fallback_c: Int = 3) -> (Bool, Tensor):
    var pair = read_jpeg(path)
    var ok = pair[0]
    var t  = pair[1].copy()
    if ok:
        return (True, t)

    var h = fallback_h
    if h <= 0:
        h = 64
    var w = fallback_w
    if w <= 0:
        w = 64
    var c = fallback_c
    if not (c == 1 or c == 3):
        c = 3

    var dummy = _make_dummy_u8_hwc(h, w, c)
    return (False, dummy)


# --------------------------- helpers: RGBA(HWC) → RGB(HWC) ---------------------------
# src: packed HWC u8, 4 channels; dst: packed HWC u8, 3 channels
fn _rgba_to_rgb_hwc(src: UnsafePointer[UInt8], w: Int, h: Int, dst: UnsafePointer[UInt8]):
    var npix = w * h
    var i = 0
    var si = 0   # source index (4 per pixel)
    var di = 0   # dest index (3 per pixel)
    while i < npix:
        # Copy R,G,B; skip A
        dst[di + 0] = src[si + 0]
        dst[di + 1] = src[si + 1]
        dst[di + 2] = src[si + 2]
        si += 4
        di += 3
        i += 1

# ---------------------------- Public API: Write ---------------------------
fn write_jpeg(path: String, t_in: Tensor, quality: Int = 90) -> Bool:
    # Clamp quality
    var q = quality
    if q < 1: q = 1
    if q > 100: q = 100

    # Ensure packed HWC/u8
    var t = t_in.copy()
    if not t.is_contiguous_hwc_u8():
        t = t.copy_to_packed_hwc()

    var w = t.width()
    var h = t.height()
    if w <= 0 or h <= 0:
        print("[write_jpeg] FAIL: invalid size ", w, "x", h)
        return False

    var dt = t.dtype()
    if dt.id != 0:
        print("[write_jpeg] FAIL: dtype id must be 0 (UInt8). got=", dt.id)
        return False

    var ch = t.channels()
    var src_ptr = t.data()
    var used_scratch = False
    var scratch_rgb = UnsafePointer[UInt8]()

    # RGBA -> RGB (drop alpha)
    if ch == 4:
        var npix = w * h
        var out_len = npix * 3
        scratch_rgb = UnsafePointer[UInt8].alloc(out_len)
        var i = 0
        var si = 0
        var di = 0
        while i < npix:
            scratch_rgb[di + 0] = src_ptr[si + 0]
            scratch_rgb[di + 1] = src_ptr[si + 1]
            scratch_rgb[di + 2] = src_ptr[si + 2]
            si += 4
            di += 3
            i += 1
        src_ptr = scratch_rgb
        ch = 3
        used_scratch = True
    else:
        if not (ch == 1 or ch == 3):
            print("[write_jpeg] FAIL: unsupported channels=", ch)
            return False

    # Growable encode buffer
    var max_size = Int(262144)
    var out_buf  = UnsafePointer[UInt8].alloc(max_size)
    var used = 0
    var ok_enc = False

    var tries = 0
    while tries < 6:
        var res = encode_jpeg(src_ptr, w, h, out_buf, max_size, ch, q)
        ok_enc = res[0]
        used   = res[1]
        if ok_enc: break
        print("[write_jpeg] grow buffer ", max_size, " -> ", max_size * 2)
        UnsafePointer[UInt8].free(out_buf)
        max_size = max_size * 2
        out_buf  = UnsafePointer[UInt8].alloc(max_size)
        tries += 1

    if used_scratch:
        UnsafePointer[UInt8].free(scratch_rgb)

    if not ok_enc:
        print("[write_jpeg] FAIL: encode_jpeg failed")
        UnsafePointer[UInt8].free(out_buf)
        return False

    # Sanitize path and write
    var p0 = _trim_spaces(path)
    if len(p0) == 0:
        p0 = String("output.jpg")
    var ok = _write_file_raw(p0, out_buf, used)

    # If "wb" fails on این رانتایم، _write_file_raw خودش fallback به "w" دارد.
    if not ok:
        # Try basename
        var base = _basename(p0)
        var p1 = _trim_spaces(base)
        if len(p1) == 0: p1 = String("output.jpg")
        if p1 != p0:
            print("[write_jpeg] WARN: retry basename '", p1, "'")
            ok = _write_file_raw(p1, out_buf, used)
    if not ok:
        var p2 = String("momijo_out.jpg")
        print("[write_jpeg] WARN: retry fallback '", p2, "'")
        ok = _write_file_raw(p2, out_buf, used)

    if ok:
        print("[write_jpeg] OK: wrote ", used, " bytes.")
    else:
        print("[write_jpeg] FAIL: all write attempts failed.")

    UnsafePointer[UInt8].free(out_buf)
    return ok





# ---------------------------- Image wrappers ----------------------------

fn read_jpeg_image(path: String) -> Image:
    var res = read_jpeg(path)
    var ok  = res[0]
    var t   = res[1].copy()
    assert(ok, "read_jpeg failed: " + path)

    var c = t.channels()
    var meta = ImageMeta()
    if c == 1:
        meta = meta.with_colorspace(ColorSpace.GRAY())
    else:
        meta = meta.with_colorspace(ColorSpace.SRGB())

    # Image(meta, tensor) per project struct signature
    return Image(meta.copy(), t.copy())

fn write_jpeg_image(path: String, img: Image, quality: Int = 90) -> Bool:
    var t = img.tensor()
    return write_jpeg(path, t, quality)
