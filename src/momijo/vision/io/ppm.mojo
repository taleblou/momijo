# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/ppm.mojo
# Description: Minimal PPM (P6) read/write without external OS deps.

from momijo.vision.image import Image
from momijo.vision.transforms.array import full
from momijo.vision.io.file_io import _read_file_bytes, _write_file_bytes

# ------------------------------------------------------------
# Small helpers (byte-level)
# ------------------------------------------------------------

fn _is_space(b: UInt8) -> Bool:
    return (b == UInt8(9)) or (b == UInt8(10)) or (b == UInt8(11)) or (b == UInt8(12)) or (b == UInt8(13)) or (b == UInt8(32))

fn _skip_spaces(data: List[UInt8], mut i: Int):
    var n = len(data)
    while i < n and _is_space(data[i]):
        i += 1

fn _skip_comment_lines(data: List[UInt8], mut i: Int):
    var n = len(data)
    while i < n and data[i] == UInt8(35):  # '#'
        while i < n and data[i] != UInt8(10):  # '\n'
            i += 1
        if i < n:
            i += 1

fn _read_token(data: List[UInt8], mut i: Int) -> String:
    _skip_spaces(data, i)
    _skip_comment_lines(data, i)
    _skip_spaces(data, i)
    var n = len(data)
    var out = String()
    while i < n:
        var b = data[i]
        if b == UInt8(35) or _is_space(b):  # '#' or whitespace
            break
        var buf = UnsafePointer[UInt8].alloc(1)
        buf[0] = b
        out = out + String(buf, 1)
        UnsafePointer[UInt8].free(buf)
        i += 1
    return out.copy()

fn _parse_int(s: String) -> Int:
    var i = 0
    var n = s.__len__()
    var sign = 1
    if n > 0 and (s[0] == '-' or s[0] == '+'):
        if s[0] == '-':
            sign = -1
        i += 1

    var acc: Int = 0

    var zero = 0
    try:
        zero = Int('0')
    except _:
        return 0  # fallback if '0' can't be converted

    while i < n:
        var ch = s[i]
        var digit = 0
        try:
            digit = Int(ch) - zero
        except _:
            break

        if digit < 0 or digit > 9:
            break

        acc = acc * 10 + digit
        i += 1

    return sign * acc

fn _append_uint_ascii(mut out: List[UInt8], val: Int):
    if val == 0:
        out.append(UInt8(48))
        return
    var x = val
    var tmp = List[UInt8]()
    while x > 0:
        var d = x % 10
        tmp.append(UInt8(48 + d))
        x = x // 10
    var j = tmp.__len__() - 1
    while j >= 0:
        out.append(tmp[j])
        j -= 1

# ------------------------------------------------------------
# Public API: read/write PPM (P6)
# ------------------------------------------------------------

fn read_ppm(path: String) -> Image:
    # _read_file_bytes returns (ok, bytes)
    var req = _read_file_bytes(path)
    var ok=req[0]
    var bytes=req[1].copy()
    if not ok or len(bytes) < 11:
        return full((32, 32, 3), UInt8(127))

    var i = 0
    var magic = _read_token(bytes, i)
    if not (magic.__len__() == 2 and magic[0] == 'P' and magic[1] == '6'):
        return full((32, 32, 3), UInt8(127))

    var w_s = _read_token(bytes, i)
    var h_s = _read_token(bytes, i)
    var max_s = _read_token(bytes, i)
    var w = _parse_int(w_s)
    var h = _parse_int(h_s)
    var maxv = _parse_int(max_s)
    if w <= 0 or h <= 0 or maxv <= 0:
        return full((32, 32, 3), UInt8(127))

    _skip_spaces(bytes, i)

    var want = w * h * 3
    var left = len(bytes) - i
    if left < want:
        return full((w, h, 3), UInt8(127))

    # Allocate packed HWC U8 image
    var img = full((h, w, 3), UInt8(0))
    var t = img.tensor()
    var ptr = t.data()           # underlying contiguous buffer (U8)

    var k = 0
    while k < want:
        ptr[k] = bytes[i + k]
        k += 1

    return img.copy()

fn write_ppm(path: String, img: Image) -> Bool:
    var w = img.width()
    var h = img.height()

    var all = List[UInt8]()
    # "P6\n"
    all.append(UInt8(80))   # 'P'
    all.append(UInt8(54))   # '6'
    all.append(UInt8(10))   # '\n'
    # "<w> <h>\n"
    _append_uint_ascii(all, w)
    all.append(UInt8(32))   # ' '
    _append_uint_ascii(all, h)
    all.append(UInt8(10))   # '\n'
    # "255\n"
    all.append(UInt8(50))   # '2'
    all.append(UInt8(53))   # '5'
    all.append(UInt8(53))   # '5'
    all.append(UInt8(10))   # '\n'

    var t = img.tensor()
    var ptr = t.data()
    var n = w * h * 3
    var i = 0
    while i < n:
        all.append(ptr[i])
        i += 1

    return _write_file_bytes(path, all)
