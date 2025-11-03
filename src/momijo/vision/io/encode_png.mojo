# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/encode_png.mojo
# Description: PNG encoder Ultra-Pro with:
#   - Gray/GA/RGB/RGBA (+Indexed) ; Gray/Indexed 1/2/4/8; non-indexed 8/16
#   - Adam7 full
#   - Smart filter (fixed-Huffman bitcost per-row)
#   - Deflate: Stored / Fixed / Dynamic
#   - LZ77 hash-chain + lazy

from collections.list import List

@always_inline
fn _be32_from_int(x: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8((x >> 24) & 255))
    out.append(UInt8((x >> 16) & 255))
    out.append(UInt8((x >> 8) & 255))
    out.append(UInt8(x & 255))
    return out.copy()


@always_inline
fn _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b

fn crc32(bytes: List[UInt8], start: Int, end: Int) -> Int:
    var crc = 0xFFFFFFFF
    var i = start
    while i < end:
        var c = Int(bytes[i])
        crc = crc ^ c
        var k = 0
        while k < 8:
            var mask = -(crc & 1)
            crc = (crc >> 1) ^ (0xEDB88320 & mask)
            k += 1
        i += 1
    return crc ^ 0xFFFFFFFF



# ---- Bit packers ---- #
fn _pack_bits_from_u8(values: List[UInt8], bits: Int, count: Int) -> List[UInt8]:
    var out = List[UInt8]()
    var acc = 0; var abits = 0; var i = 0
    while i < count:
        var v = Int(values[i]) & ((1 << bits) - 1)
        acc = (acc << bits) | v; abits += bits
        while abits >= 8:
            var b = (acc >> (abits - 8)) & 255
            out.append(UInt8(b)); abits -= 8
        i += 1
    if abits > 0:
        var b2 = (acc << (8 - abits)) & 255
        out.append(UInt8(b2))
    return out.copy()

fn _u8_to_u16_be(buf: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    var i = 0
    while i < len(buf):
        var v = Int(buf[i]) * 257
        out.append(UInt8((v >> 8) & 255)); out.append(UInt8(v & 255))
        i += 1
    return out.copy()

# ---- Deflate helpers ---- #
struct BitWriter:
    var buf: List[UInt8]
    var cur: Int
    var nbits: Int
    fn __init__(out self):
        self.buf = List[UInt8](); self.cur = 0; self.nbits = 0
    fn write_bits(mut self, v: Int, n: Int) -> Bool:
        var i = 0
        while i < n:
            var b = (v >> i) & 1
            self.cur |= (b << self.nbits); self.nbits += 1
            if self.nbits == 8:
                self.buf.append(UInt8(self.cur & 255)); self.cur = 0; self.nbits = 0
            i += 1
        return True
    fn align_byte(mut self) -> None:
        if self.nbits > 0: self.buf.append(UInt8(self.cur & 255)); self.cur = 0; self.nbits = 0

fn _zlib_header(mut outList: List[UInt8]) -> None:
    var CMF = 0x78; var FLG = 0; var t = 0
    while t < 256:
        if ((CMF << 8) + t) % 31 == 0: FLG = t; break
        t += 1
    outList.append(UInt8(CMF)); outList.append(UInt8(FLG))

fn _zlib_footer(data: List[UInt8], mut outList: List[UInt8]) -> None:
    var ad = adler32(data); var adbe = _be32_from_int(ad)
    outList.append(adbe[0]); outList.append(adbe[1]); outList.append(adbe[2]); outList.append(adbe[3])

# Fixed Huffman writers
fn _put_fixed_litlen(mut bw: BitWriter, sym: Int) -> Bool:
    if sym >= 0 and sym <= 143: return bw.write_bits(sym + 48, 8)
    elif sym >= 144 and sym <= 255: return bw.write_bits(sym + 256, 9)
    elif sym >= 256 and sym <= 279: return bw.write_bits(sym - 256, 7)
    elif sym >= 280 and sym <= 287: return bw.write_bits(sym - 88, 8)
    else: return False
fn _put_fixed_dist(mut bw: BitWriter, sym: Int) -> Bool: return bw.write_bits(sym, 5)




fn _len_to_code(length: Int) -> (Int, Int, Int):
    var LEN_BASE = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
    var LEN_EXTRA = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
    var li = 0
    while li < len(LEN_BASE):
        var base = LEN_BASE[li]
        var next = 258 if li == len(LEN_BASE)-1 else LEN_BASE[li+1]
        var max_here = next - 1
        if length >= base and length <= max_here:
            var eb = LEN_EXTRA[li]
            return (li + 257, eb, length - base)
        li += 1
    return (285, 0, 0)

fn _dist_to_code(dist: Int) -> (Int, Int, Int):
    var DIST_BASE = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    var DIST_EXTRA = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14]
    var di = 0
    while di < len(DIST_BASE):
        var base = DIST_BASE[di]
        var next = 32768 if di == len(DIST_BASE)-1 else DIST_BASE[di+1]
        var max_d = next - 1
        if dist >= base and dist <= max_d:
            var eb = DIST_EXTRA[di]
            return (di, eb, dist - base)
        di += 1
    return (0, 0, 0)

# Hash-chain + lazy
fn _hash3(a: Int, b: Int, c: Int) -> Int: return ((a * 911 + b * 131 + c * 31) & 0xFFFF)
struct HC:
    var head: List[Int]; var prev: List[Int]
    fn __init__(out self, n: Int):
        self.head = List[Int](); self.prev = List[Int]()
        var i = 0;
        while i < 65536: self.head.append(-1); i += 1
        i = 0;
        while i < n: self.prev.append(-1); i += 1
fn _insert(mut hc: HC, data: List[UInt8], pos: Int) -> None:
    if pos + 2 >= len(data): return
    var h = _hash3(Int(data[pos]), Int(data[pos+1]), Int(data[pos+2]))
    hc.prev[pos] = hc.head[h]; hc.head[h] = pos
fn _find_best(hc: HC, data: List[UInt8], pos: Int, wnd: Int, max_len: Int, chain_cap: Int) -> (Int, Int):
    var best_len = 0; var best_dist = 0
    var max_back = pos if pos < wnd else wnd
    var cur = -1
    if pos + 2 < len(data): cur = hc.head[_hash3(Int(data[pos]), Int(data[pos+1]), Int(data[pos+2]))]
    var chain = 0
    while cur >= 0 and chain < chain_cap and (pos - cur) <= max_back:
        var j = 0
        while j < max_len and pos + j < len(data) and data[cur + j] == data[pos + j]: j += 1
        if j >= 3 and j > best_len: best_len = j; best_dist = pos - cur;
        if best_len == max_len: break
        cur = hc.prev[cur]; chain += 1
    return (best_len, best_dist)


# ---- Tok: explicit Copyable & Movable + ctors ----
struct Tok(Copyable, Movable):
    var lit: Int
    var length: Int
    var dist: Int
    var len_code: Int
    var len_ebits: Int
    var len_extra: Int
    var dist_code: Int
    var dist_ebits: Int
    var dist_extra: Int
    var is_lit: Bool

    # main constructor
    fn __init__(out self,
                lit: Int, length: Int, dist: Int,
                len_code: Int, len_ebits: Int, len_extra: Int,
                dist_code: Int, dist_ebits: Int, dist_extra: Int,
                is_lit: Bool):
        self.lit = lit
        self.length = length
        self.dist = dist
        self.len_code = len_code
        self.len_ebits = len_ebits
        self.len_extra = len_extra
        self.dist_code = dist_code
        self.dist_ebits = dist_ebits
        self.dist_extra = dist_extra
        self.is_lit = is_lit

    # copy constructor
    fn __copyinit__(out self, other: Self):
        self.lit = other.lit
        self.length = other.length
        self.dist = other.dist
        self.len_code = other.len_code
        self.len_ebits = other.len_ebits
        self.len_extra = other.len_extra
        self.dist_code = other.dist_code
        self.dist_ebits = other.dist_ebits
        self.dist_extra = other.dist_extra
        self.is_lit = other.is_lit

@always_inline
fn make_lit(lit: Int) -> Tok:
    return Tok(lit, 0, 0, 0, 0, 0, 0, 0, 0, True)

@always_inline
fn make_match(length: Int, dist: Int,
              len_code: Int, len_ebits: Int, len_extra: Int,
              dist_code: Int, dist_ebits: Int, dist_extra: Int) -> Tok:
    return Tok(0, length, dist, len_code, len_ebits, len_extra, dist_code, dist_ebits, dist_extra, False)


fn _build_tokens_hc(data: List[UInt8]) -> List[Tok]:
    var toks = List[Tok]()
    var hc = HC(len(data))
    var pos = 0; var WND = 32768; var CHAIN = 32

    while pos < len(data):
        _insert(hc, data, pos)
        var m = _find_best(hc, data, pos, WND, 258, CHAIN)

        # lazy
        if m[0] >= 3 and pos + 1 < len(data):
            _insert(hc, data, pos + 1)
            var m2 = _find_best(hc, data, pos + 1, WND, 258, CHAIN)
            if m2[0] > m[0] + 1:
                toks.append(make_lit(Int(data[pos])))
                pos += 1
                continue

        if m[0] >= 3:
            var lc = _len_to_code(m[0])
            var dc = _dist_to_code(m[1])
            toks.append(make_match(m[0], m[1], lc[0], lc[1], lc[2], dc[0], dc[1], dc[2]))
            var j = 1
            while j < m[0]:
                _insert(hc, data, pos + j)
                j += 1
            pos += m[0]
        else:
            toks.append(make_lit(Int(data[pos])))
            pos += 1

    # end-of-block literal 256
    toks.append(make_lit(256))
    return toks.copy()


# Fixed deflate stream
fn deflate_fixed_stream(data: List[UInt8]) -> List[UInt8]:
    var bw = BitWriter()
    # BFINAL=1, BTYPE=01
    bw.write_bits(1,1); bw.write_bits(1,1); bw.write_bits(0,1)
    var toks = _build_tokens_hc(data.copy())
    var i = 0
    while i < len(toks):
        var t = toks[i].copy()
        if t.is_lit:
            _put_fixed_litlen(bw, t.lit)
        else:
            _put_fixed_litlen(bw, t.len_code)
            if t.len_ebits > 0: bw.write_bits(t.len_extra, t.len_ebits)
            _put_fixed_dist(bw, t.dist_code)
            if t.dist_ebits > 0: bw.write_bits(t.dist_extra, t.dist_ebits)
        i += 1
    bw.align_byte(); return bw.buf.copy()

# Dynamic from ultra-plus (reused here simplified)
struct Huff(Copyable, Movable):
    var first_code:   List[Int]
    var first_symbol: List[Int]
    var count:        List[Int]
    var values:       List[Int]
    var max_bits:     Int

    fn __init__(out self):
        self.first_code   = List[Int]()
        self.first_symbol = List[Int]()
        self.count        = List[Int]()
        self.values       = List[Int]()
        self.max_bits     = 0

    fn __copyinit__(out self, other: Self):
        self.first_code   = other.first_code.copy()
        self.first_symbol = other.first_symbol.copy()
        self.count        = other.count.copy()
        self.values       = other.values.copy()
        self.max_bits     = other.max_bits


fn _build_huffman(code_lengths: List[Int], max_bits: Int) -> (Bool, Huff):
    var h = Huff(); h.max_bits = max_bits
    var count = List[Int](); var i = 0;
    while i <= max_bits: count.append(0); i += 1
    var n = len(code_lengths); var sym = 0; var values = List[Int]()
    while sym < n:
        var l = code_lengths[sym]
        if l < 0 or l > max_bits: return (False, h.copy())
        if l > 0: count[l] = count[l] + 1
        sym += 1
    var first_code = List[Int](); var first_symbol = List[Int](); i = 0;
    while i <= max_bits: first_code.append(0); first_symbol.append(0); i += 1
    var code = 0; var s = 0; i = 1
    while i <= max_bits:
        first_code[i] = code; first_symbol[i] = s
        code = (code + count[i]) << 1; s += count[i]; i += 1
    var next = List[Int](); i = 0;
    while i <= max_bits: next.append(0); i += 1
    sym = 0
    while sym < n:
        var l2 = code_lengths[sym]
        if l2 > 0:
            var idx = first_symbol[l2] + next[l2]
            while len(values) <= idx: values.append(0)
            values[idx] = sym; next[l2] = next[l2] + 1
        sym += 1
    h.count = count.copy(); h.first_code = first_code.copy(); h.first_symbol = first_symbol.copy(); h.values = values.copy()
    return (True, h.copy())



struct CLE(Copyable, Movable):
    var sym:   Int
    var ebits: Int
    var extra: Int

    fn __init__(out self, sym: Int = 0, ebits: Int = 0, extra: Int = 0):
        self.sym   = sym
        self.ebits = ebits
        self.extra = extra

    fn __copyinit__(out self, other: Self):
        self.sym   = other.sym
        self.ebits = other.ebits
        self.extra = other.extra


fn _rle_len_stream(lens: List[Int]) -> List[CLE]:
    var out = List[CLE]()
    var i = 0
    while i < len(lens):
        var L = lens[i]

        if L == 0:
            # شمارش زنجیرهٔ صفرها
            var run = 1
            var j = i + 1
            while j < len(lens) and lens[j] == 0 and run < 138:
                run += 1
                j += 1

            if run >= 11:
                out.append(CLE(18, 7, run - 11))        # repeat zero 11..138
            elif run >= 3:
                out.append(CLE(17, 3, run - 3))         # repeat zero 3..10
            else:
                # 1 یا 2 صفر: خودِ 0 را بنویس
                var k = 0
                while k < run:
                    out.append(CLE(0, 0, 0))
                    k += 1

            i = j
            continue

        else:
            # طول غیرصفر
            out.append(CLE(L, 0, 0))

            # شمارش تکرار همین طول
            var run2 = 1
            var j2 = i + 1
            while j2 < len(lens) and lens[j2] == L and run2 < 6:
                run2 += 1
                j2 += 1

            if run2 >= 3:
                # کُد 16: تکرار طول قبلی 3..6
                out.append(CLE(16, 2, run2 - 3))
            elif run2 == 2:
                # برای دو بار، یک بار دیگر همان L را بنویس
                out.append(CLE(L, 0, 0))
            # اگر run2==1 کاری لازم نیست

            i = j2
            continue

        # این خط عملاً اجرا نمی‌شود، ولی برای کامل بودن حلقه:
        i += 1

    return out.copy()


fn _lens_from_hist(hist: List[Int], max_bits: Int) -> List[Int]:
    var lens = List[Int](); var i = 0
    while i < len(hist):
        var f = hist[i]; var L = 0
        if f > 0:
            if f >= 128: L = 7
            elif f >= 64: L = 8
            elif f >= 32: L = 9
            elif f >= 16: L = 10
            elif f >= 8:  L = 11
            elif f >= 4:  L = 12
            else: L = 13
            if L > max_bits: L = max_bits
            if L < 1: L = 1
        lens.append(L); i += 1
    return lens.copy()

fn _code_of(h: Huff, sym: Int, mut out_bits: Int, mut out_len:  Int) -> Bool:
    var l = 1
    while l <= h.max_bits:
        var cnt = h.count[l]
        if cnt == 0: l += 1; continue
        var first = h.first_code[l]; var base = h.first_symbol[l]; var i = 0
        while i < cnt:
            if h.values[base + i] == sym:
                out_bits = first + i; out_len = l; return True
            i += 1
        l += 1
    return False

fn deflate_dynamic_stream(data: List[UInt8]) -> List[UInt8]:
    # 1) Build tokens & histograms
    var toks = _build_tokens_hc(data)
    var lit = List[Int](); var i = 0
    while i < 286: lit.append(0); i += 1
    var dist = List[Int](); i = 0
    while i < 30: dist.append(0); i += 1

    var k = 0
    while k < len(toks):
        var t = toks[k].copy()
        if t.is_lit:
            lit[t.lit] = lit[t.lit] + 1
        else:
            lit[t.len_code]   = lit[t.len_code] + 1
            dist[t.dist_code] = dist[t.dist_code] + 1
        k += 1

    if lit[256] == 0: lit[256] = 1

    var last_lit = 285
    while last_lit >= 257 and lit[last_lit] == 0: last_lit -= 1
    var last_dist = 29
    while last_dist >= 0 and dist[last_dist] == 0: last_dist -= 1
    if last_dist < 0: last_dist = 0

    var HLIT  = last_lit + 1 - 257;
    if HLIT  < 0: HLIT  = 0
    var HDIST = last_dist + 1;
    if HDIST < 1: HDIST = 1

    var lit_len  = _lens_from_hist(lit,  15)
    var dist_len = _lens_from_hist(dist, 15)

    var ii = last_lit + 1
    while ii < len(lit_len):  lit_len[ii]  = 0; ii += 1
    ii = last_dist + 1
    while ii < len(dist_len): dist_len[ii] = 0; ii += 1

    var ORDER = [16,17,18, 0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]

    var cl_len = List[Int](); var ci = 0
    while ci < 19: cl_len.append(0); ci += 1
    cl_len[0] = 3; cl_len[16] = 3; cl_len[17] = 3; cl_len[18] = 3

    var last_cl = 18
    while last_cl >= 0 and cl_len[ORDER[last_cl]] == 0: last_cl -= 1
    var HCLEN = last_cl + 1 - 4;
    if HCLEN < 0: HCLEN = 0

    var okcl  = _build_huffman(cl_len, 7)
    var cltab = okcl[1].copy()

    var lens_all = List[Int]()
    var a = 0
    while a < (257 + HLIT): lens_all.append(lit_len[a]); a += 1
    var b = 0
    while b < HDIST: lens_all.append(dist_len[b]); b += 1

    var rle = _rle_len_stream(lens_all)

    var litok    = _build_huffman(lit_len,  15)
    var distok   = _build_huffman(dist_len, 15)
    var lit_tab  = litok[1].copy()
    var dist_tab = distok[1].copy()

    # 2) zlib header + dynamic block
    var out = List[UInt8]()
    _zlib_header(out)

    var bw = BitWriter()
    # BFINAL=1, BTYPE=10
    bw.write_bits(1,1); bw.write_bits(0,1); bw.write_bits(1,1)

    bw.write_bits(HLIT, 5)
    bw.write_bits(HDIST - 1, 5)
    bw.write_bits(HCLEN, 4)

    var q = 0
    while q < 4 + HCLEN:
        bw.write_bits(cl_len[ORDER[q]], 3)
        q += 1

    # 3) emit code-length RLE
    var ri = 0
    while ri < len(rle):
        var e = rle[ri].copy()
        put_code(bw, cltab, e.sym)
        if e.sym == 16:      bw.write_bits(e.extra, 2)
        elif e.sym == 17:    bw.write_bits(e.extra, 3)
        elif e.sym == 18:    bw.write_bits(e.extra, 7)
        ri += 1

    # 4) emit tokens
    var ti = 0
    while ti < len(toks):
        var t = toks[ti].copy()
        if t.is_lit:
            put_code(bw, lit_tab, t.lit)
        else:
            put_code(bw, lit_tab, t.len_code)
            if t.len_ebits  > 0: bw.write_bits(t.len_extra,  t.len_ebits)
            put_code(bw, dist_tab, t.dist_code)
            if t.dist_ebits > 0: bw.write_bits(t.dist_extra, t.dist_ebits)
        ti += 1

    bw.align_byte()

    var iout = 0
    while iout < len(bw.buf):
        out.append(bw.buf[iout])
        iout += 1

    _zlib_footer(data, out)
    return out.copy()


fn adler32(data: List[UInt8]) -> Int:
    var MOD = 65521
    var a = 1
    var b = 0
    var i = 0
    while i < len(data):
        a = (a + Int(data[i])) % MOD
        b = (b + a) % MOD
        i += 1
    return (b << 16) | a

fn _le16(x: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8(x & 255))
    out.append(UInt8((x >> 8) & 255))
    return out.copy()

fn _le32(x: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8(x & 255))
    out.append(UInt8((x >> 8) & 255))
    out.append(UInt8((x >> 16) & 255))
    out.append(UInt8((x >> 24) & 255))
    return out.copy()

fn zlib_stored(data: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()


    var CMF = 0x78
    var FLG = 0
    var t = 0
    while t < 256:
        if ((CMF << 8) + t) % 31 == 0:
            FLG = t
            break
        t += 1
    out.append(UInt8(CMF))
    out.append(UInt8(FLG))

    # بلوک‌های Stored حداکثر 65535 بایت
    var pos = 0
    var n = len(data)
    while pos < n:
        var left = n - pos
        var blk = left
        if blk > 65535:
            blk = 65535
        var bfinal = 0
        if pos + blk >= n:
            bfinal = 1

        # هدر بلوک: BFINAL + BTYPE=00  (فقط بایت صفر یا یک)
        out.append(UInt8(bfinal))

        # LEN و NLEN (کوچک‌-اندیان)
        var le  = _le16(blk)
        var nle = _le16(blk ^ 0xFFFF)
        out.append(le[0]);  out.append(le[1])
        out.append(nle[0]); out.append(nle[1])

        # payload
        var k = 0
        while k < blk:
            out.append(data[pos + k])
            k += 1
        pos += blk
    # Adler32 در انتهای zlib به صورت big-endian نوشته می‌شود
    var ad = adler32(data)
    var adle = _le32(ad)
    out.append(adle[3]); out.append(adle[2]); out.append(adle[1]); out.append(adle[0])

    return out.copy()

fn zlib_deflate(data: List[UInt8], mode: Int) -> List[UInt8]:
    if mode == 2: return deflate_dynamic_stream(data.copy())
    elif mode == 1:
        var out = List[UInt8](); _zlib_header(out)
        var defl = deflate_fixed_stream(data.copy())
        var i = 0;
        while i < len(defl): out.append(defl[i]); i += 1
        _zlib_footer(data, out); return out.copy()
    else: return zlib_stored(data.copy())

# ---- Filter scoring (fixed-Huffman bit cost) ---- #
fn _fixed_code_len(sym: Int) -> Int:
    if sym >= 0 and sym <= 143: return 8
    elif sym >= 144 and sym <= 255: return 9
    elif sym >= 256 and sym <= 279: return 7
    elif sym >= 280 and sym <= 287: return 8
    return 15

fn _bitcost_fixed(bytes: List[UInt8]) -> Int:
    # cost of encoding bytes as literals under fixed huffman + end-of-block
    var hist = List[Int](); var i = 0;
    while i < 288: hist.append(0); i += 1
    i = 0;
    while i < len(bytes): hist[Int(bytes[i])] = hist[Int(bytes[i])] + 1; i += 1
    hist[256] = hist[256] + 1
    var cost = 0; i = 0;
    while i < 288:
        var c = hist[i]
        if c > 0: cost += c * _fixed_code_len(i)
        i += 1
    return cost

@always_inline
fn _paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = p - a;
    if pa < 0: pa = -pa
    var pb = p - b;
    if pb < 0: pb = -pb
    var pc = p - c;
    if pc < 0: pc = -pc
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c

fn _filter_row(filter_type: Int, prev: List[UInt8], cur: List[UInt8], bpp: Int) -> List[UInt8]:
    var out = List[UInt8](); out.append(UInt8(filter_type))
    var x = 0
    while x < len(cur):
        var raw = Int(cur[x])
        var left = 0; var up = 0; var ul = 0
        if x >= bpp: left = Int(cur[x - bpp])
        if len(prev) > 0:
            up = Int(prev[x])
            if x >= bpp: ul = Int(prev[x - bpp])
        var val = 0
        if filter_type == 0: val = raw
        elif filter_type == 1: val = (raw - left) & 255
        elif filter_type == 2: val = (raw - up) & 255
        elif filter_type == 3: val = (raw - ((left + up) >> 1)) & 255
        else: val = (raw - _paeth(left, up, ul)) & 255
        out.append(UInt8(val)); x += 1
    return out.copy()

fn _build_scan_filtered(width: Int, height: Int, bytes_per_sample: Int, channels: Int, data: List[UInt8], filter_mode: Int) -> List[UInt8]:
    var out = List[UInt8]()
    var stride = width * channels * bytes_per_sample
    var y = 0
    # print("[scan-simple] begin W=", width, " H=", height, " C=", channels, " Bps=", bytes_per_sample, " stride=", stride, " data.len=", len(data))
    while y < height:
        out.append(UInt8(0))  # filter type = None
        var base = y * stride
        var x = 0
        while x < stride:
            out.append(data[base + x])
            x += 1
        # دیباگ: اولین سه کانال ردیف
        if stride >= 3:
            var r0 = out[len(out)-stride]
            var g0 = out[len(out)-stride+1]
            var b0 = out[len(out)-stride+2]
            # print("[scan-simple] y=", y, " first_px=", r0, ",", g0, ",", b0)
        y += 1
    return out.copy()

# ---- Adam7 pass extraction (encoder) ---- #
fn _extract_adam7_pass(width: Int, height: Int, bytes_per_sample: Int, channels: Int, data: List[UInt8], pass_idx: Int) -> (Int, Int, List[UInt8]):
    var XOFF = [0,4,0,2,0,1,0]; var YOFF = [0,0,4,0,2,0,1]
    var XSP  = [8,8,4,4,2,2,1]; var YSP  = [8,8,8,4,4,2,2]
    var x0 = XOFF[pass_idx]; var y0 = YOFF[pass_idx]
    var xs = XSP[pass_idx]; var ys = YSP[pass_idx]
    var pw = 0; var ph = 0
    if width > x0: pw = _ceil_div(width - x0, xs)
    if height > y0: ph = _ceil_div(height - y0, ys)
    if pw <= 0 or ph <= 0: return (0,0,List[UInt8]())
    var out = List[UInt8]()
    var py = 0
    while py < ph:
        var px = 0
        while px < pw:
            var fy = y0 + py * ys; var fx = x0 + px * xs
            var src = (fy * width + fx) * channels * bytes_per_sample
            var k = 0
            while k < channels * bytes_per_sample:
                out.append(data[src + k]); k += 1
            px += 1
        py += 1
    return (pw, ph, out.copy())

# ---- Chunk helper ---- #
# ---- Chunk helper (fixed) ----
fn _append_chunk(mut outList: List[UInt8],
                 typ: (UInt8,UInt8,UInt8,UInt8),
                 payload: List[UInt8]) -> None:
    # Length (big-endian)
    var L = _be32_from_int(len(payload))
    outList.append(L[0]); outList.append(L[1]); outList.append(L[2]); outList.append(L[3])

    # Type
    outList.append(typ[0]); outList.append(typ[1]); outList.append(typ[2]); outList.append(typ[3])

    # Payload
    var i = 0
    while i < len(payload):
        outList.append(payload[i])
        i += 1

    # CRC over [type+payload]
    var crc_start = len(outList) - len(payload) - 4
    var crc_val = crc32(outList, crc_start, len(outList))
    var C = _be32_from_int(crc_val)
    outList.append(C[0]); outList.append(C[1]); outList.append(C[2]); outList.append(C[3])



# ---- Main encode ---- #
fn png_from_hwc_u8(width: Int, height: Int, channels: Int, data: List[UInt8],
                   interlace: Int = 0, compress: Int = 0, filter_mode: Int = 0,
                   palette_mode: Int = 0, max_colors: Int = 256, bit_depth_out: Int = 8) -> (Bool, List[UInt8]):
    # چک‌های اولیه
    if width <= 0 or height <= 0: return (False, List[UInt8]())
    if channels < 1 or channels > 4: return (False, List[UInt8]())
    if len(data) != width * height * channels: return (False, List[UInt8]())


    var bit_depth = 8
    var ctype = 2   # فرض: RGB
    if channels == 1: ctype = 0  # Gray
    elif channels == 3: ctype = 2 # RGB
    elif channels == 4: ctype = 6 # RGBA
    elif channels == 2: ctype = 4 # Gray+Alpha

    var bytes_per_sample = 1
    var bpp = channels * bytes_per_sample

    # print("[png] begin(simple) W=", width, " H=", height, " C=", channels, " data.len=", len(data))
    # print("[png] mode= direct  ctype=", ctype, " bit_depth=", bit_depth, " bpp=", bpp, " interlace=0 filter=0 zlib=stored")

    var scan = _build_scan_filtered(width, height, bytes_per_sample, channels, data.copy(), 0)
    # print("[png] scan.len=", len(scan))

    # ساخت خروجی PNG
    var out = List[UInt8]()
    out.extend([UInt8(0x89),UInt8(0x50),UInt8(0x4E),UInt8(0x47),UInt8(0x0D),UInt8(0x0A),UInt8(0x1A),UInt8(0x0A)])

    # IHDR
    var ih = List[UInt8]()
    var wb = _be32_from_int(width); var hb = _be32_from_int(height)
    ih.append(wb[0]); ih.append(wb[1]); ih.append(wb[2]); ih.append(wb[3])
    ih.append(hb[0]); ih.append(hb[1]); ih.append(hb[2]); ih.append(hb[3])
    ih.append(UInt8(bit_depth)); ih.append(UInt8(ctype))
    ih.append(UInt8(0)); ih.append(UInt8(0)); ih.append(UInt8(0))  # interlace=0
    _append_chunk(out, (UInt8(0x49),UInt8(0x48),UInt8(0x44),UInt8(0x52)), ih.copy())

    # IDAT با zlib STORED
    var z = zlib_stored(scan.copy())
    _append_chunk(out, (UInt8(0x49),UInt8(0x44),UInt8(0x41),UInt8(0x54)), z.copy())

    # IEND
    _append_chunk(out, (UInt8(0x49),UInt8(0x45),UInt8(0x4E),UInt8(0x44)), List[UInt8]())

    # print("[png] done(simple) out.len=", len(out))
    return (True, out.copy())


# helper to emit a symbol using a canonical table (no nested functions)
@always_inline
fn put_code(mut bw: BitWriter, tab: Huff, sym: Int) -> None:
    var bits = 0
    var ln = 0
    var ok = _code_of(tab, sym, bits, ln)
    if ok:
        bw.write_bits(bits, ln)
