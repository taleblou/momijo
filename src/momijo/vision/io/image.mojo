# MIT License 
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/io/image.mojo

from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType
from momijo.vision.image import Image, ColorSpace, ImageMeta
from momijo.vision.transforms.array import full
# from momijo.vision.io.ppm import read_ppm, write_ppm
from momijo.vision.io.png import read_png, write_png, has_png_codec
# from momijo.vision.io.jpeg import read_jpeg, write_jpeg, has_jpeg_codec
from stdlib.pathlib import Path
from stdlib.os import mkdir, stat, ENOENT
# -------------------------------------------------------------------------
# Primary API (delegates to registry implementation)
# -------------------------------------------------------------------------

fn read_image(path: String) -> (Bool, Tensor):
    var img = read_image_any(path)
    return (True, img.tensor())

# -------------------------------------------------------------------------
# Fallback helpers
# -------------------------------------------------------------------------

fn make_dummy_u8_hwc_tensor(h: Int, w: Int, c: Int) -> Tensor:
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c
    var buf = UnsafePointer[UInt8].alloc(n)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                var v = (y + x + ch * 32) % 256
                buf[y * s0 + x * s1 + ch * s2] = UInt8(v)
                ch += 1
            x += 1
        y += 1
    return Tensor(buf, n, h, w, c, s0, s1, s2, DType.UInt8)

fn read_image_with_fallback(path: String, fallback_h: Int = 64, fallback_w: Int = 64, fallback_c: Int = 3) -> (Bool, Tensor):
    var (ok, t) = read_image(path)
    if ok: return (True, t)
    return (False, make_dummy_u8_hwc_tensor(fallback_h, fallback_w, fallback_c))

fn ensure_outdir(path: String) -> Bool:
    var p = Path(path)
    var parent = p.parent()
    if parent.to_string() == "":
        return True  # No parent → assume current dir

    try:
        var info = stat(parent)
        return True  # Exists
    except err:
        if err.errno() == ENOENT:
            try:
                return mkdir(parent, recursive=True)
            except _:
                return False
        else:
            return False

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

fn _to_lower(s: String) -> String:
    var out = String()
    var i = 0

    while i < s.__len__():
        var ch = s[i]
        try:
            var code = Int(ch)
            var upper_a = Int('A')
            var upper_z = Int('Z')
            var delta = Int('a') - Int('A')

            if code >= upper_a and code <= upper_z:
                # تبدیل دستی به lowercase: با آرایه تک‌بایتی
                var b = UnsafePointer[UInt8].alloc(1)
                b[0] = UInt8(code + delta)
                var low = String(b, 1)
                UnsafePointer[UInt8].free(b)
                out = out + low
            else:
                out = out + ch  # ch is StringSlice
        except _:
            out = out + ch
        i += 1

    return out.copy()



fn unsafe_char_from_code(code: Int) -> String:
    var b = UnsafePointer[UInt8].alloc(1)
    b[0] = UInt8(code)
    var s = String(b, 1)
    UnsafePointer[UInt8].free(b)
    return s


fn _ends_with(s: String, suffix: String) -> Bool:
    var n = s.__len__()
    var m = suffix.__len__()
    if m > n: return False
    var i = 0
    while i < m:
        if s[n - m + i] != suffix[i]:
            return False
        i += 1
    return True

# -------------------------------------------------------------------------
# Format dispatch
# -------------------------------------------------------------------------

fn read_image_any(path: String) -> Image:
    var p = _to_lower(path)

    # if _ends_with(p, ".ppm"):
    #     return read_ppm(path)

    if _ends_with(p, ".png"):
        if has_png_codec():
            var req = read_png(path)
            var ok=req[0]
            var tensor=req[1].copy()
            if ok:
                var meta = ImageMeta().with_colorspace(ColorSpace.SRGB())
                return Image(meta.copy(),tensor.copy())
            else:
                return full((128, 128, 3), UInt8(127))
        else:
            return full((128, 128, 3), UInt8(127))

    # if _ends_with(p, ".jpg") or _ends_with(p, ".jpeg"):
    #     if has_jpeg_codec():
    #         var req = read_jpeg(path)
    #         var okj=req[0]
    #         var tj=req[1].copy()
    #         if okj:
    #             var meta_j = ImageMeta().with_colorspace(ColorSpace.SRGB())
    #             return Image(meta_j.copy(),tj.copy())
    #         else:
    #             return full((160, 120, 3), UInt8(127))
    #     else:
    #         return full((160, 120, 3), UInt8(127))

    return full((32, 32, 3), UInt8(127))

fn write_image_any(path: String, img: Image) -> Bool: 
    
    var p = _to_lower(path)    
    var ii = img.ensure_packed_hwc_u8(True)

    # if _ends_with(p, ".ppm"):
    #     return write_ppm(path, img)

    if _ends_with(p, ".png"):
        # if has_png_codec():
            return write_png(path, ii.tensor())
        # else:
        #     var alt = path + String(".ppm")
        #     return write_ppm(alt, img)

    # if _ends_with(p, ".jpg") or _ends_with(p, ".jpeg"): 
    #     # if has_jpeg_codec():
    #         return write_jpeg(path, img.tensor())
    #     # else:
    #     #     var alt = path + String(".ppm")
    #     #     return write_ppm(alt, img)

    # #var alt2 = path + String(".ppm")
    return False #rite_ppm(path, img)
