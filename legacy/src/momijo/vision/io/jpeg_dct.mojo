@always_inline
fn _descale(x: Int, s: Int) -> Int:
    # Arithmetic right shift with symmetric rounding-to-nearest.
    var add = 1 << (s - 1)
    if x < 0:
        return -(((-x) + add) >> s)
    return (x + add) >> s

@always_inline
fn _mul_fix(a: Int, b: Int) -> Int:
    # a * (b / 8192) with symmetric rounding
    return _descale(a * b, 13)

# ---------- DCT constants (scaled by 2^13 = 8192) ----------
fn _FIX_0_298631336() -> Int: return 2446
fn _FIX_0_390180644() -> Int: return 3196
fn _FIX_0_541196100() -> Int: return 4433
fn _FIX_0_765366865() -> Int: return 6270
fn _FIX_0_899976223() -> Int: return 7373
fn _FIX_1_175875602() -> Int: return 9633
fn _FIX_1_501321110() -> Int: return 12299
fn _FIX_1_847759065() -> Int: return 15137
fn _FIX_1_961570560() -> Int: return 16069
fn _FIX_2_053119869() -> Int: return 16819
fn _FIX_2_562915447() -> Int: return 20995
fn _FIX_3_072711026() -> Int: return 25172
fn _FIX_0_707106781() -> Int: return 5793   # sqrt(1/2)

# Optional aliases (as در کدت بود)
fn _FIX_0_382683433_alt() -> Int: return 3135
fn _FIX_1_306562965_alt() -> Int: return 10703

# ---------- 1-D FDCT on 8 samples (row transform, in-place over tmp[row*8..row*8+7]) ----------
@always_inline
fn _fdct_1d_row(tmp: UnsafePointer[Int], row: Int):
    var i0 = row * 8

    var d0 = tmp[i0 + 0]
    var d1 = tmp[i0 + 1]
    var d2 = tmp[i0 + 2]
    var d3 = tmp[i0 + 3]
    var d4 = tmp[i0 + 4]
    var d5 = tmp[i0 + 5]
    var d6 = tmp[i0 + 6]
    var d7 = tmp[i0 + 7]

    # Even part
    var tmp0 = d0 + d7
    var tmp7 = d0 - d7
    var tmp1 = d1 + d6
    var tmp6 = d1 - d6
    var tmp2 = d2 + d5
    var tmp5 = d2 - d5
    var tmp3 = d3 + d4
    var tmp4 = d3 - d4

    var tmp10 = tmp0 + tmp3
    var tmp11 = tmp1 + tmp2
    var tmp12 = tmp1 - tmp2
    var tmp13 = tmp0 - tmp3

    # DC/AC4
    tmp[i0 + 0] = tmp10 + tmp11
    tmp[i0 + 4] = tmp10 - tmp11

    # AC2/AC6 (correct AAN form)
    tmp[i0 + 2] = _mul_fix(tmp12 + tmp13, _FIX_0_707106781())
    tmp[i0 + 6] = _mul_fix(tmp13 - tmp12, _FIX_0_707106781())

    # Odd part (AAN integer, 13-bit)
    var z10 = tmp4 + tmp5
    var z11 = tmp5 + tmp6
    var z12 = tmp6 + tmp7
    var z13 = tmp4 + tmp6
    var z14 = tmp5 + tmp7

    var z5  = _mul_fix(z13 + z14, _FIX_1_175875602())

    # Pre-scale individual terms
    tmp4 = _mul_fix(tmp4, _FIX_0_298631336())
    tmp5 = _mul_fix(tmp5, _FIX_2_053119869())
    tmp6 = _mul_fix(tmp6, _FIX_3_072711026())
    tmp7 = _mul_fix(tmp7, _FIX_1_501321110())

    z13 = _mul_fix(z13, -_FIX_1_961570560()) + z5
    z14 = _mul_fix(z14, -_FIX_0_390180644()) + z5
    z10 = _mul_fix(z10, -_FIX_0_899976223())
    z11 = _mul_fix(z11, -_FIX_2_562915447())

    tmp[i0 + 7] = tmp4 + z10 + z13
    tmp[i0 + 5] = tmp5 + z11 + z14
    tmp[i0 + 3] = tmp6 + z11 + z13
    tmp[i0 + 1] = tmp7 + z10 + z14

# ---------- 1-D FDCT on columns (reads tmp rows, writes to dst) ----------
@always_inline
fn _fdct_1d_col(tmp: UnsafePointer[Int], col: Int, dst: UnsafePointer[Int]):
    var d0 = tmp[0 * 8 + col]
    var d1 = tmp[1 * 8 + col]
    var d2 = tmp[2 * 8 + col]
    var d3 = tmp[3 * 8 + col]
    var d4 = tmp[4 * 8 + col]
    var d5 = tmp[5 * 8 + col]
    var d6 = tmp[6 * 8 + col]
    var d7 = tmp[7 * 8 + col]

    # Even part
    var tmp0 = d0 + d7
    var tmp7 = d0 - d7
    var tmp1 = d1 + d6
    var tmp6 = d1 - d6
    var tmp2 = d2 + d5
    var tmp5 = d2 - d5
    var tmp3 = d3 + d4
    var tmp4 = d3 - d4

    var tmp10 = tmp0 + tmp3
    var tmp11 = tmp1 + tmp2
    var tmp12 = tmp1 - tmp2
    var tmp13 = tmp0 - tmp3

    var out0 = tmp10 + tmp11
    var out4 = tmp10 - tmp11
    var out2 = _mul_fix(tmp12 + tmp13, _FIX_0_707106781())
    var out6 = _mul_fix(tmp13 - tmp12, _FIX_0_707106781())

    # Odd part (mirror of row)
    var z10 = tmp4 + tmp5
    var z11 = tmp5 + tmp6
    var z12 = tmp6 + tmp7
    var z13 = tmp4 + tmp6
    var z14 = tmp5 + tmp7

    var z5  = _mul_fix(z13 + z14, _FIX_1_175875602())

    tmp4 = _mul_fix(tmp4, _FIX_0_298631336())
    tmp5 = _mul_fix(tmp5, _FIX_2_053119869())
    tmp6 = _mul_fix(tmp6, _FIX_3_072711026())
    tmp7 = _mul_fix(tmp7, _FIX_1_501321110())

    z13 = _mul_fix(z13, -_FIX_1_961570560()) + z5
    z14 = _mul_fix(z14, -_FIX_0_390180644()) + z5
    z10 = _mul_fix(z10, -_FIX_0_899976223())
    z11 = _mul_fix(z11, -_FIX_2_562915447())

    var out7 = tmp4 + z10 + z13
    var out5 = tmp5 + z11 + z14
    var out3 = tmp6 + z11 + z13
    var out1 = tmp7 + z10 + z14

    dst[0 * 8 + col] = out0
    dst[4 * 8 + col] = out4
    dst[2 * 8 + col] = out2
    dst[6 * 8 + col] = out6
    dst[7 * 8 + col] = out7
    dst[5 * 8 + col] = out5
    dst[3 * 8 + col] = out3
    dst[1 * 8 + col] = out1

# ---------- Public API ----------
# src: pointer to the top-left pixel of an 8x8 window (UInt8, single channel)
# stride: byte distance between adjacent rows
# dst: 64 Int coefficients (natural order)
fn dct_8x8(src: UnsafePointer[UInt8], stride: Int, dst: UnsafePointer[Int]):
    # Load with level-shift (0..255 -> -128..127)
    var tmp = UnsafePointer[Int].alloc(64)
    var y = 0
    while y < 8:
        var rowp = src + y * stride
        var x = 0
        while x < 8:
            tmp[y * 8 + x] = Int(rowp[x]) - 128
            x += 1
        y += 1

    var r = 0
    while r < 8:
        _fdct_1d_row(tmp, r)
        r += 1

    var c = 0
    while c < 8:
        _fdct_1d_col(tmp, c, dst)
        c += 1

    UnsafePointer[Int].free(tmp)
