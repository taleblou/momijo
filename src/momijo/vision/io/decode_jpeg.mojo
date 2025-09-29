# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/decode_jpeg.mojo

from momijo.vision.io.jpeg_decoder import is_jpeg, parse_segment
from momijo.vision.io.bitreader import BitReader
from momijo.vision.io.huffman import HuffmanTable
from momijo.vision.io.jpeg_scan import decode_block
from momijo.vision.io.jpeg_color_ycbcr import merge_ycbcr_to_rgb

from momijo.vision.tensor import Tensor
from momijo.vision.image import Image, ImageMeta, ColorSpace
from momijo.vision.transforms.array import full

# --- Marker helpers as functions (avoid file-scope const/let) ---
fn _SOF0() -> Int: return 0xFFC0
fn _DQT()  -> Int: return 0xFFDB
fn _DHT()  -> Int: return 0xFFC4
fn _SOS()  -> Int: return 0xFFDA
fn _EOI()  -> Int: return 0xFFD9

# --- Minimal fallback image (1x1 black RGB) ---
fn _fallback_image() -> Image:
    var t = full((1, 1, 3), UInt8(0)).tensor()
    var m = ImageMeta().with_colorspace(ColorSpace.SRGB())
    return Image(t, m)

fn decode_jpeg(ptr: UnsafePointer[UInt8], length: Int) -> (Bool, Image):
    # Quick guards
    if length < 4:
        return (False, _fallback_image())
    if not is_jpeg(ptr):
        return (False, _fallback_image())

    # End pointer used by parse_segment
    var end = ptr + length

    # Cursor pointer and numeric offset (offset helps compute scan length)
    var cursor = ptr + 2            # skip SOI (0xFFD8)
    var cur_off: Int = 2

    var width: Int = 0
    var height: Int = 0

    # Quantization table pointer (points inside the stream; minimal single-table path)
    var quant: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(1)
    var have_quant = False

    # Huffman tables
    var dc_table = HuffmanTable()
    var ac_table = HuffmanTable()

    # Scan (entropy) region
    var scan_ptr: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(1)
    var scan_off: Int = 0
    var scan_len: Int = 0
    var have_scan = False

    var is_color = False

    # --- Parse segments until SOS/EOI; collect SOF0/DQT/DHT and locate scan ---
    while cursor < end:
        var res = parse_segment(cursor, end)
        if not res.ok:
            print("[JPEG] Truncated/invalid segment; aborting")
            break

        var marker = res.seg.marker

        if marker == _SOF0():
            # SOF0 payload: [P] [Yh] [Yl] [Xh] [Xl] [Nf] ...
            height = (Int(res.seg.data_ptr[1]) << 8) | Int(res.seg.data_ptr[2])
            width  = (Int(res.seg.data_ptr[3]) << 8) | Int(res.seg.data_ptr[4])
            var num_components = Int(res.seg.data_ptr[5])
            is_color = (num_components == 3)

        else:
            if marker == _DQT():
                # Minimal path: assume a single 8-bit table; skip Pq/Tq (first byte)
                quant = res.seg.data_ptr + 1
                have_quant = True

            else:
                if marker == _DHT():
                    # Layout: [TcTh] [16 counts] [symbols...]
                    var is_dc = (Int(res.seg.data_ptr[0]) & 0x10) == 0
                    var counts = res.seg.data_ptr + 1
                    var symbols = counts + 16
                    if is_dc:
                        dc_table.build(counts, symbols)
                    else:
                        ac_table.build(counts, symbols)

                else:
                    if marker == _SOS():
                        # After SOS header, entropy-coded data follows until EOI
                        scan_ptr = cursor + 2 + res.seg.length
                        scan_off = cur_off + 2 + res.seg.length
                        scan_len = 0
                        have_scan = True

                    else:
                        if marker == _EOI():
                            # Close the scan span using numeric offsets
                            if have_scan:
                                scan_len = cur_off - scan_off
                            break

        # Advance: marker (2 bytes) + payload length
        var adv = 2 + res.seg.length
        cursor = cursor + adv
        cur_off = cur_off + adv

    # Validate collected information
    if width <= 0 or height <= 0 or (not have_quant) or (not have_scan) or scan_len <= 0:
        return (False, _fallback_image())

    # --- Entropy decode ---
    # If BitReader lacks a (ptr, len) initializer in your codebase, replace the next line with:
    #   var br = BitReader(); br_init(br, scan_ptr, scan_len)
    var br = BitReader(scan_ptr, scan_len)

    var block = UnsafePointer[Int].alloc(64)

    if is_color:
        # Decode Y, Cb, Cr (no subsampling; baseline 8x8 blocks)
        var npix = width * height
        var buf_y  = UnsafePointer[UInt8].alloc(npix)
        var buf_cb = UnsafePointer[UInt8].alloc(npix)
        var buf_cr = UnsafePointer[UInt8].alloc(npix)

        var yb = 0
        var prev_dc_y = 0
        var prev_dc_cb = 0
        var prev_dc_cr = 0
        while yb < height:
            var xb = 0
            while xb < width:
                # Y block
                prev_dc_y = decode_block(br, dc_table, ac_table, quant, prev_dc_y, block)
                var by = 0
                while by < 8:
                    var bx = 0
                    while bx < 8:
                        var xx = xb + bx
                        var yy = yb + by
                        if xx < width and yy < height:
                            var v = block[by * 8 + bx] + 128
                            if v < 0: v = 0
                            if v > 255: v = 255
                            buf_y[yy * width + xx] = UInt8(v)
                        bx = bx + 1
                    by = by + 1

                # Cb block
                prev_dc_cb = decode_block(br, dc_table, ac_table, quant, prev_dc_cb, block)
                by = 0
                while by < 8:
                    var bx2 = 0
                    while bx2 < 8:
                        var xx2 = xb + bx2
                        var yy2 = yb + by
                        if xx2 < width and yy2 < height:
                            var v2 = block[by * 8 + bx2] + 128
                            if v2 < 0: v2 = 0
                            if v2 > 255: v2 = 255
                            buf_cb[yy2 * width + xx2] = UInt8(v2)
                        bx2 = bx2 + 1
                    by = by + 1

                # Cr block
                prev_dc_cr = decode_block(br, dc_table, ac_table, quant, prev_dc_cr, block)
                by = 0
                while by < 8:
                    var bx3 = 0
                    while bx3 < 8:
                        var xx3 = xb + bx3
                        var yy3 = yb + by
                        if xx3 < width and yy3 < height:
                            var v3 = block[by * 8 + bx3] + 128
                            if v3 < 0: v3 = 0
                            if v3 > 255: v3 = 255
                            buf_cr[yy3 * width + xx3] = UInt8(v3)
                        bx3 = bx3 + 1
                    by = by + 1

                xb = xb + 8
            yb = yb + 8

        # Convert planar YCbCr to interleaved RGB
        var rgb_bytes = width * height * 3
        var rgb_buf = UnsafePointer[UInt8].alloc(rgb_bytes)
        merge_ycbcr_to_rgb(buf_y, buf_cb, buf_cr, rgb_buf, width, height)

        # Pack into HWC tensor
        var img = full((height, width, 3), UInt8(0))
        var t = img.tensor()
        var dst = t.data()
        var i = 0
        while i < rgb_bytes:
            dst[i] = rgb_buf[i]
            i = i + 1

        var m = ImageMeta().with_colorspace(ColorSpace.SRGB())
        return (True, Image(t, m))

    else:
        # Grayscale path
        var npix_g = width * height
        var buf = UnsafePointer[UInt8].alloc(npix_g)

        var yb2 = 0
        var prev_dc = 0
        while yb2 < height:
            var xb2 = 0
            while xb2 < width:
                prev_dc = decode_block(br, dc_table, ac_table, quant, prev_dc, block)
                var by2 = 0
                while by2 < 8:
                    var bx4 = 0
                    while bx4 < 8:
                        var xx = xb2 + bx4
                        var yy = yb2 + by2
                        if xx < width and yy < height:
                            var v = block[by2 * 8 + bx4] + 128
                            if v < 0: v = 0
                            if v > 255: v = 255
                            buf[yy * width + xx] = UInt8(v)
                        bx4 = bx4 + 1
                    by2 = by2 + 1
                xb2 = xb2 + 8
            yb2 = yb2 + 8

        var img_g = full((height, width, 1), UInt8(0))
        var tg = img_g.tensor()
        var dstg = tg.data()
        var k = 0
        while k < npix_g:
            dstg[k] = buf[k]
            k = k + 1

        # Use sRGB meta if a dedicated Gray() colorspace is not available
        var mg = ImageMeta().with_colorspace(ColorSpace.SRGB())
        return (True, Image(tg, mg))
