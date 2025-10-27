# MIT License
# SPDX-License-Identifier: MIT
# File: src/momijo/vision/io/png.mojo

from collections.list import List
from momijo.vision.image import Image, ColorSpace, Layout
from momijo.vision.io.file_io import read_all_bytes,write_all_bytes
from momijo.vision.io.encode_png import png_from_hwc_u8
from momijo.vision.io.decode_png import decode_png_bytes, decode_png_bytes_u16


fn _to_hwc_u8(packed: Image) -> (Int, Int, Int, List[UInt8]):
    var H = packed.height()
    var W = packed.width()
    var C = packed.channels()
    if H <= 0 or W <= 0 or C < 1 or C > 4:
        return (0, 0, 0, List[UInt8]())

    # Guard: prevent insane allocations (helps catch accidental bad dims)
    var total = H * W * C
    if total < 0 or total > 1_000_000_000:
        # unrealistic; abort safely
        return (0, 0, 0, List[UInt8]())

    var out = List[UInt8]()
    out.reserve(total)

    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var k = 0
            while k < C:
                out.append(packed.get_u8(y, x, k))   # 0=R,1=G,2=B
                k += 1
            x += 1
        y += 1

    return (H, W, C, out.copy())  # ← نیازی به .copy() نیست




fn _from_hwc_u8(H: Int, W: Int, C: Int, buf: List[UInt8]) -> Image:
    var img = Image.new_hwc_u8(H, W, C, UInt8(0), ColorSpace.SRGB(), Layout.HWC())
    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var k = 0
            var idx = (y * W + x) * C
            while k < C:
                img.set_u8(y, x, k, buf[idx + k])
                k += 1
            x += 1
        y += 1
    return img.copy()


# --- helpers: no nested functions ---

@always_inline
fn _buf_at_hwc(buf: List[UInt8], W: Int, C: Int, y: Int, x: Int, k: Int) -> UInt8:
    var idx = (y * W + x) * C + k
    return buf[idx]

@always_inline
fn _safe_ch(C: Int, val: UInt8) -> UInt8:
    # tiny helper just to keep branches simple if channel doesn't exist
    return val

# --- main debug printer: no nested functions ---

@always_inline
fn debug_sample_rgb(H: Int, W: Int, C: Int, buf: List[UInt8], tag: String):
    if H == 0 or W == 0 or C == 0:
        print("[dbg]", tag, " empty")
        return

    var ymid = H // 2
    var xmid = W // 2

    # TL
    var tl0 = _buf_at_hwc(buf, W, C, 0, 0, 0)
    var tl1 = UInt8(0)
    var tl2 = UInt8(0)
    if C > 1: tl1 = _buf_at_hwc(buf, W, C, 0, 0, 1)
    if C > 2: tl2 = _buf_at_hwc(buf, W, C, 0, 0, 2)

    # MID
    var m0 = _buf_at_hwc(buf, W, C, ymid, xmid, 0)
    var m1 = UInt8(0)
    var m2 = UInt8(0)
    if C > 1: m1 = _buf_at_hwc(buf, W, C, ymid, xmid, 1)
    if C > 2: m2 = _buf_at_hwc(buf, W, C, ymid, xmid, 2)

    # BR
    var br0 = _buf_at_hwc(buf, W, C, H - 1, W - 1, 0)
    var br1 = UInt8(0)
    var br2 = UInt8(0)
    if C > 1: br1 = _buf_at_hwc(buf, W, C, H - 1, W - 1, 1)
    if C > 2: br2 = _buf_at_hwc(buf, W, C, H - 1, W - 1, 2)

    print("[dbg]", tag, " C=", C,
          " TL=", tl0, ",", tl1, ",", tl2,
          " MID=", m0, ",", m1, ",", m2,
          " BR=", br0, ",", br1, ",", br2)




# Encode
fn write_png(path: String, img: Image, interlace: Int = 0, compress: Int = 2, filter_mode: Int = -1, palette_mode: Int = 0, max_colors: Int = 256, bit_depth_out: Int = 8) raises -> Bool:
    # var (rr,rg,rb) = img.get_rgb_u8(0,0)
    # print("[chk image] TL=", rr, ",", rg, ",", rb)
    var dims = _to_hwc_u8(img.copy())
    var H = dims[0]; var W = dims[1]; var C = dims[2]; var buf = dims[3].copy() 
    # debug_sample_rgb(H, W, C, buf, "pre-encode")
    if H <= 0 or W <= 0 or C < 1 or C > 4: return False
    var ok_bytes = png_from_hwc_u8(W, H, C, buf.copy(), 0, compress, 0, palette_mode, max_colors, bit_depth_out)
    if not ok_bytes[0]: return False
    return write_all_bytes(path, ok_bytes[1].copy())

# Decode (u8)
fn read_png(path: String) raises -> (Bool, Image):
    var bytes = read_all_bytes(path)
    var dec = decode_png_bytes(bytes.copy())
    var ok = dec[0]
    if not ok:
        return (False, Image.new_hwc_u8(0,0,3,UInt8(0)))
    var w = dec[1]; var h = dec[2]; var c = dec[3]; var buf = dec[4].copy()
    var img = _from_hwc_u8(h, w, c, buf.copy())
    return (True, img.copy())

# Raw u16 decode (non-indexed, 16-bit PNGs): returns channels and a flat List[UInt16]
fn read_png_u16_raw(path: String) raises -> (Bool, Int, Int, Int, List[UInt16]):
    var bytes = read_all_bytes(path)
    var dec = decode_png_bytes_u16(bytes.copy())
    return (dec[0], dec[1], dec[2], dec[3], dec[4].copy())