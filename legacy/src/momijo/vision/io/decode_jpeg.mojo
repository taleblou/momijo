# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/vision/io/decode_jpeg.mojo
# Description: Minimal baseline JPEG decoder (no subsampling) using single DQT/DHT,
#              BitReader with stuffing handling, and planar -> HWC packing.
#              Supports grayscale (1 component) and 3-component YCbCr 4:4:4.

from momijo.vision.io.jpeg_decoder import is_jpeg
from momijo.vision.io.jpeg_decoder import parse_segment    # returns a struct with .ok, .seg.marker, .seg.length, .seg.data_ptr
from momijo.vision.io.bitreader import BitReader
from momijo.vision.io.huffman import HuffmanTable
from momijo.vision.io.jpeg_scan import decode_block       # (br, dc_table, ac_table, qtab, prev_dc, out_block[64]) -> Int (new prev_dc)
from momijo.vision.io.jpeg_color_ycbcr import merge_ycbcr_to_rgb

from momijo.vision.transforms.array import full
from momijo.vision.tensor import Tensor
from momijo.vision.image import Image, ImageMeta, ColorSpace

# ---------------- Marker helpers ----------------
@always_inline
fn _SOI() -> Int: return 0xFFD8
@always_inline
fn _EOI() -> Int: return 0xFFD9
@always_inline
fn _SOF0() -> Int: return 0xFFC0
@always_inline
fn _DQT()  -> Int: return 0xFFDB
@always_inline
fn _DHT()  -> Int: return 0xFFC4
@always_inline
fn _SOS()  -> Int: return 0xFFDA

# ---------------- Fallback image ----------------
@always_inline
fn _fallback_image() -> Image:
    # 1x1x3 black, packed HWC u8
    var arr = full((1, 1, 3), UInt8(0))
    var t = arr.tensor()
    var meta = ImageMeta().with_colorspace(ColorSpace.SRGB())
    return Image(meta.copy(), t.copy())

# ---------------- Utility: scan to end of entropy-coded segment ----------------
# Starting at 'scan_ptr' (first entropy-coded byte after SOS header), walk forward
# through stuffed-bytes (0xFF 0x00) until we hit a marker boundary: 0xFF followed
# by a non-zero byte. Return (ok, length, marker_low) where marker_low is the
# non-zero byte following 0xFF (e.g., 0xD9 for EOI).
fn _find_scan_end(scan_ptr: UnsafePointer[UInt8], end_ptr: UnsafePointer[UInt8]) -> (Bool, Int, UInt8):
    var p = scan_ptr
    while p < end_ptr:
        var b = p[0]
        p = p + 1
        if b == UInt8(0xFF):
            if p >= end_ptr:
                # 0xFF at the very end: treat as boundary without a following code
                return (True, Int(scan_ptr.distance_to(p - 1)), UInt8(0x00))
            var nextb = p[0]
            if nextb == UInt8(0x00):
                # stuffed 0xFF -> consume the 0x00 and continue
                p = p + 1
                continue
            # boundary: marker starts at 0xFF nextb (do not include 0xFF in scan length)
            var length = Int(scan_ptr.distance_to(p - 1))
            return (True, length, nextb)
    # reached end without explicit marker; treat whole tail as scan
    return (True, Int(scan_ptr.distance_to(end_ptr)), UInt8(0x00))

# ---------------- Main decoder ----------------
fn decode_jpeg(ptr: UnsafePointer[UInt8], length: Int) -> (Bool, Image):
    # Basic guards
    if length < 4:
        return (False, _fallback_image())
    if not is_jpeg(ptr):
        return (False, _fallback_image())

    var end = ptr + length
    var cursor = ptr

    # Expect SOI
    if length < 2:
        return (False, _fallback_image())
    # Skip SOI (0xFFD8)
    cursor = cursor + 2

    # Parsed state
    var width: Int = 0
    var height: Int = 0
    var num_components: Int = 0
    var is_color = False

    # Minimal single-table path: keep only the first DQT and a single pair of DHT (DC+AC).
    # Note: pointers reference inside the bytestream; do not free them.
    var qtab_ptr: UnsafePointer[UInt8] = UnsafePointer[UInt8]()  # 64 entries (8-bit)
    var have_qtab = False

    var dc_table = HuffmanTable()
    var ac_table = HuffmanTable()
    var have_dc = False
    var have_ac = False

    # Locate SOS and compute scan span (start pointer and length)
    var scan_ptr = UnsafePointer[UInt8]()
    var scan_len: Int = 0
    var have_scan = False

    # ------------- Parse segments until SOS -------------
    while cursor < end:
        var seg = parse_segment(cursor, end)
        if not seg.ok:
            # malformed segment
            return (False, _fallback_image())

        var marker = seg.seg.marker
        # Advance to after marker + payload (2 bytes for marker already accounted by parse_segment contract)
        # parse_segment is assumed to return 'cursor' still at marker start; we move by (2 + payload)
        var adv = 2 + seg.seg.length

        if marker == _SOF0():
            # SOF0 payload: [P] [Yh][Yl] [Xh][Xl] [Nf] [...]
            if seg.seg.length < 8:
                return (False, _fallback_image())
            var dp = seg.seg.data_ptr
            height = (Int(dp[1]) << 8) | Int(dp[2])
            width  = (Int(dp[3]) << 8) | Int(dp[4])
            num_components = Int(dp[5])
            is_color = (num_components == 3)

        elif marker == _DQT():
            # Minimal path: pick the first 8-bit table and keep its 64 entries
            if seg.seg.length < 1 + 64:
                # allow truncated but present; still invalid for decoding
                return (False, _fallback_image())
            var dp = seg.seg.data_ptr
            var pq_tq = dp[0]  # Pq(1b)<<4 | Tq(4b)
            var pq = (pq_tq >> 4) & UInt8(0x0F)
            if pq != UInt8(0):  # only 8-bit supported in minimal path
                return (False, _fallback_image())
            qtab_ptr = dp + 1         # first of 64 bytes
            have_qtab = True

        elif marker == _DHT():
            # Layout: [TcTh] [16 counts] [symbols...]
            # Minimal path: retain the first DC and first AC table encountered.
            if seg.seg.length < 1 + 16:
                return (False, _fallback_image())
            var dp = seg.seg.data_ptr
            var tc_th = dp[0]
            var is_dc = ((Int(tc_th) & 0x10) == 0)  # Tc=0 -> DC, Tc=1 -> AC
            var counts = dp + 1
            var symbols = counts + 16
            if is_dc and not have_dc:
                dc_table.build(counts, symbols)
                have_dc = True
            elif (not is_dc) and not have_ac:
                ac_table.build(counts, symbols)
                have_ac = True
            # Otherwise ignore additional tables in the minimal path.

        elif marker == _SOS():
            # After SOS header, entropy-coded scan begins immediately and runs until
            # the next marker (0xFF followed by non-zero). We compute the span now.
            if seg.seg.length < 6:
                return (False, _fallback_image())
            var after_sos = cursor + adv
            var ok_span = _find_scan_end(after_sos, end)
            if not ok_span[0]:
                return (False, _fallback_image())
            scan_ptr = after_sos
            scan_len = ok_span[1]
            have_scan = True
            # We can stop parsing headers now; EOI (or another marker) follows after the scan.
            cursor = after_sos + scan_len
            break

        elif marker == _EOI():
            # Unexpected EOI before SOS
            break

        # advance to next segment
        cursor = cursor + adv

    # Validate essentials
    if width <= 0 or height <= 0:
        return (False, _fallback_image())
    if not have_qtab or not have_dc or not have_ac or not have_scan or scan_len <= 0:
        return (False, _fallback_image())

    # ------------- Entropy decode -------------
    var br = BitReader(scan_ptr, scan_len)

    # Working block buffer (IDCT output or dequantized domain as produced by decode_block)
    var block = UnsafePointer[Int].alloc(64)

    if is_color:
        # Planar buffers
        var npix = width * height
        var buf_y  = UnsafePointer[UInt8].alloc(npix)
        var buf_cb = UnsafePointer[UInt8].alloc(npix)
        var buf_cr = UnsafePointer[UInt8].alloc(npix)

        var prev_dc_y = 0
        var prev_dc_cb = 0
        var prev_dc_cr = 0

        var by_row = 0
        while by_row < height:
            var bx_col = 0
            while bx_col < width:
                # Y
                prev_dc_y = decode_block(br, dc_table, ac_table, qtab_ptr, prev_dc_y, block)
                var yy = 0
                while yy < 8:
                    var xx = 0
                    while xx < 8:
                        var X = bx_col + xx
                        var Y = by_row + yy
                        if X < width and Y < height:
                            var v = block[yy * 8 + xx] + 128
                            if v < 0: v = 0
                            if v > 255: v = 255
                            buf_y[Y * width + X] = UInt8(v)
                        xx += 1
                    yy += 1

                # Cb
                prev_dc_cb = decode_block(br, dc_table, ac_table, qtab_ptr, prev_dc_cb, block)
                yy = 0
                while yy < 8:
                    var xx2 = 0
                    while xx2 < 8:
                        var X2 = bx_col + xx2
                        var Y2 = by_row + yy
                        if X2 < width and Y2 < height:
                            var v2 = block[yy * 8 + xx2] + 128
                            if v2 < 0: v2 = 0
                            if v2 > 255: v2 = 255
                            buf_cb[Y2 * width + X2] = UInt8(v2)
                        xx2 += 1
                    yy += 1

                # Cr
                prev_dc_cr = decode_block(br, dc_table, ac_table, qtab_ptr, prev_dc_cr, block)
                yy = 0
                while yy < 8:
                    var xx3 = 0
                    while xx3 < 8:
                        var X3 = bx_col + xx3
                        var Y3 = by_row + yy
                        if X3 < width and Y3 < height:
                            var v3 = block[yy * 8 + xx3] + 128
                            if v3 < 0: v3 = 0
                            if v3 > 255: v3 = 255
                            buf_cr[Y3 * width + X3] = UInt8(v3)
                        xx3 += 1
                    yy += 1

                bx_col += 8
            by_row += 8

        # Interleave to RGB (HWC)
        var total_rgb = width * height * 3
        var rgb_buf = UnsafePointer[UInt8].alloc(total_rgb)
        merge_ycbcr_to_rgb(buf_y, buf_cb, buf_cr, rgb_buf, width, height)

        var arr_rgb = full((height, width, 3), UInt8(0))
        var t_rgb = arr_rgb.tensor()
        var dst = t_rgb.data()
        var i = 0
        while i < total_rgb:
            dst[i] = rgb_buf[i]
            i += 1

        UnsafePointer[UInt8].free(rgb_buf)
        UnsafePointer[UInt8].free(buf_y)
        UnsafePointer[UInt8].free(buf_cb)
        UnsafePointer[UInt8].free(buf_cr)
        UnsafePointer[Int].free(block)

        var meta = ImageMeta().with_colorspace(ColorSpace.SRGB())
        return (True, Image(meta.copy(), t_rgb.copy()))

    else:
        # Grayscale
        var npix_g = width * height
        var buf_g = UnsafePointer[UInt8].alloc(npix_g)

        var prev_dc = 0
        var by_row2 = 0
        while by_row2 < height:
            var bx_col2 = 0
            while bx_col2 < width:
                prev_dc = decode_block(br, dc_table, ac_table, qtab_ptr, prev_dc, block)
                var yy2 = 0
                while yy2 < 8:
                    var xx4 = 0
                    while xx4 < 8:
                        var X = bx_col2 + xx4
                        var Y = by_row2 + yy2
                        if X < width and Y < height:
                            var v = block[yy2 * 8 + xx4] + 128
                            if v < 0: v = 0
                            if v > 255: v = 255
                            buf_g[Y * width + X] = UInt8(v)
                        xx4 += 1
                    yy2 += 1
                bx_col2 += 8
            by_row2 += 8

        var arr_g = full((height, width, 1), UInt8(0))
        var t_g = arr_g.tensor()
        var dstg = t_g.data()
        var k = 0
        while k < npix_g:
            dstg[k] = buf_g[k]
            k += 1

        UnsafePointer[UInt8].free(buf_g)
        UnsafePointer[Int].free(block)

        # If no dedicated Gray() exists, reuse sRGB meta (convention in project)
        var meta_g = ImageMeta().with_colorspace(ColorSpace.SRGB())
        return (True, Image(meta_g.copy(), t_g.copy()))
