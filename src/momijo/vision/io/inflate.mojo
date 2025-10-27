# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/inflate.mojo
# Description: zlib inflate: Stored (BTYPE=0), Fixed (BTYPE=1), Dynamic (BTYPE=2)

from collections.list import List

struct BitReader:
    var data: List[UInt8]
    var pos: Int
    var bit: Int

    fn __init__(out self, src: List[UInt8]):
        self.data = src.copy()
        self.pos = 0
        self.bit = 0

    fn read_bits(mut self, nbits: Int) -> (Bool, Int):
        var v = 0
        var n = 0
        while n < nbits:
            if self.pos >= len(self.data):
                return (False, 0)
            var cur = Int(self.data[self.pos])
            var b = (cur >> self.bit) & 1
            v |= (b << n)
            self.bit += 1
            if self.bit == 8:
                self.bit = 0
                self.pos += 1
            n += 1
        return (True, v)

    fn align_to_byte(mut self) -> None:
        if self.bit != 0:
            self.bit = 0
            self.pos += 1

@always_inline
fn _le16(b0: UInt8, b1: UInt8) -> Int:
    return Int(b0) | (Int(b1) << 8)

# ---------------- Canonical Huffman ---------------- #

struct Huffman(Copyable, Movable):
    var first_code:   List[Int]
    var first_symbol: List[Int]
    var max_bits:     Int
    var count:        List[Int]   # count[bits]
    var values:       List[Int]   # symbols in canonical order

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
        self.max_bits     = other.max_bits   # <-- بدون .copy()

# Build canonical decoder from code lengths
fn _build_huffman(code_lengths: List[Int], max_bits: Int) -> (Bool, Huffman):
    var h = Huffman()
    h.max_bits = max_bits

    var count = List[Int]()
    var i = 0
    while i <= max_bits:
        count.append(0)
        i += 1

    var n = len(code_lengths)
    var sym = 0
    var values = List[Int]()
    while sym < n:
        var l = code_lengths[sym]
        if l < 0 or l > max_bits:
            return (False, h.copy())
        if l > 0:
            count[l] = count[l] + 1
        sym += 1

    var first_code = List[Int]()
    var first_symbol = List[Int]()
    i = 0
    while i <= max_bits:
        first_code.append(0)
        first_symbol.append(0)
        i += 1

    var code = 0
    var s = 0
    i = 1
    while i <= max_bits:
        first_code[i] = code
        first_symbol[i] = s
        code = (code + count[i]) << 1
        s += count[i]
        i += 1

    var next = List[Int]()
    i = 0
    while i <= max_bits:
        next.append(0)
        i += 1

    sym = 0
    while sym < n:
        var l2 = code_lengths[sym]
        if l2 > 0:
            var idx = first_symbol[l2] + next[l2]
            while len(values) <= idx:
                values.append(0)
            values[idx] = sym
            next[l2] = next[l2] + 1
        sym += 1

    h.count        = count.copy()
    h.first_code   = first_code.copy()
    h.first_symbol = first_symbol.copy()
    h.values       = values.copy()
    return (True, h.copy())

# Decode a symbol using canonical Huffman tables
fn _huff_decode(mut br: BitReader, h: Huffman) -> (Bool, Int):
    var code = 0
    var bitlen = 0
    while bitlen < h.max_bits:
        var rb = br.read_bits(1)
        if not rb[0]:
            return (False, 0)
        code |= (rb[1] << bitlen)
        bitlen += 1

        var count = h.count[bitlen]
        if count == 0:
            continue

        var first = h.first_code[bitlen]
        if code < first:
            continue

        var idx = h.first_symbol[bitlen] + (code - first)
        if idx < 0 or idx >= len(h.values):
            continue

        return (True, h.values[idx])
    return (False, 0)

# Fixed lit/len & dist tables
fn _build_fixed_tables() -> (Huffman, Huffman):
    var litlen_len = List[Int]()
    var i = 0
    while i < 288:
        var l = 0
        if i <= 143:      l = 8
        elif i <= 255:    l = 9
        elif i <= 279:    l = 7
        else:             l = 8
        litlen_len.append(l)
        i += 1
    var dist_len = List[Int]()
    i = 0
    while i < 32:
        dist_len.append(5)
        i += 1
    var ok1 = _build_huffman(litlen_len, 9)
    var ok2 = _build_huffman(dist_len, 5)
    return (ok1[1].copy(), ok2[1].copy())

fn _read_code_lengths(mut br: BitReader, n: Int, cl_table: Huffman) -> (Bool, List[Int]):
    var out = List[Int]()
    var prev = 0
    var i = 0
    while i < n:
        var ok_sym = _huff_decode(br, cl_table)
        if not ok_sym[0]: return (False, out.copy())
        var sym = ok_sym[1].copy()
        if sym <= 15:
            out.append(sym)
            prev = sym
            i += 1
        elif sym == 16:
            var okb = br.read_bits(2)
            if not okb[0]: return (False, out.copy())
            var repeat = 3 + okb[1].copy()
            var k = 0
            while k < repeat and i < n:
                out.append(prev)
                k += 1; i += 1
        elif sym == 17:
            var okb2 = br.read_bits(3)
            if not okb2[0]: return (False, out.copy())
            var repeat2 = 3 + okb2[1].copy()
            var k2 = 0
            while k2 < repeat2 and i < n:
                out.append(0)
                k2 += 1; i += 1
            prev = 0
        elif sym == 18:
            var okb3 = br.read_bits(7)
            if not okb3[0]: return (False, out.copy())
            var repeat3 = 11 + okb3[1].copy()
            var k3 = 0
            while k3 < repeat3 and i < n:
                out.append(0)
                k3 += 1; i += 1
            prev = 0
        else:
            return (False, out.copy())
    return (True, out.copy())

# Inflate one block using litlen/dist Huffman tables
fn _inflate_block_with_tables(mut br: BitReader, litlen: Huffman, dist: Huffman, mut outList: List[UInt8]) -> Bool:
    var LEN_BASE  = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
    var LEN_EXTRA = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
    var DIST_BASE = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    var DIST_EXTRA= [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14]

    while True:
        var ok_sym = _huff_decode(br, litlen)
        if not ok_sym[0]: return False
        var sym = ok_sym[1]

        if sym < 256:
            outList.append(UInt8(sym))
        elif sym == 256:
            return True
        else:
            var len_sym = sym - 257
            if len_sym < 0 or len_sym >= len(LEN_BASE): return False
            var length = LEN_BASE[len_sym]
            var extra_l = LEN_EXTRA[len_sym]
            if extra_l > 0:
                var okb = br.read_bits(extra_l)
                if not okb[0]: return False
                length += okb[1]

            var ok_d = _huff_decode(br, dist)
            if not ok_d[0]: return False
            var dcode = ok_d[1]
            if dcode < 0 or dcode >= len(DIST_BASE): return False
            var distv = DIST_BASE[dcode]
            var extra_d = DIST_EXTRA[dcode]
            if extra_d > 0:
                var okdb = br.read_bits(extra_d)
                if not okdb[0]: return False
                distv += okdb[1]

            if distv <= 0 or distv > len(outList): return False
            var i = 0
            while i < length:
                var b = outList[len(outList) - distv]
                outList.append(b)
                i += 1
    return False

# Public: zlib inflate (stored + fixed + dynamic)
fn zlib_inflate(z: List[UInt8]) -> (Bool, List[UInt8]):
    if len(z) < 6:
        return (False, List[UInt8]())
    var CMF = Int(z[0])
    var FLG = Int(z[1])
    if (CMF & 0x0F) != 8:
        return (False, List[UInt8]())

    var br = BitReader(z.copy())
    br.pos = 2; br.bit = 0

    var out = List[UInt8]()
    while True:
        if br.pos >= len(z):
            return (False, List[UInt8]())

        var ok_hdr = br.read_bits(3)
        if not ok_hdr[0]: return (False, List[UInt8]())
        var bits = ok_hdr[1]
        var bfinal = bits & 1
        var btype  = (bits >> 1) & 3

        if btype == 0:
            br.align_to_byte()
            if br.pos + 4 > len(z) - 4: return (False, List[UInt8]())
            var LEN  = _le16(z[br.pos],   z[br.pos+1])
            var NLEN = _le16(z[br.pos+2], z[br.pos+3])
            br.pos += 4
            if (LEN ^ NLEN) != 0xFFFF: return (False, List[UInt8]())
            if br.pos + LEN > len(z) - 4: return (False, List[UInt8]())
            var i = 0
            while i < LEN:
                out.append(z[br.pos + i])
                i += 1
            br.pos += LEN

        elif btype == 1:
            var fixed = _build_fixed_tables()
            if not _inflate_block_with_tables(br, fixed[0], fixed[1], out):
                return (False, List[UInt8]())

        elif btype == 2:
            var ok_h = br.read_bits(5);  
            if not ok_h[0]: return (False, List[UInt8]())
            var HLIT  = 257 + ok_h[1]
            var ok_d = br.read_bits(5);  
            if not ok_d[0]: return (False, List[UInt8]())
            var HDIST = 1 + ok_d[1]
            var ok_c = br.read_bits(4);  
            if not ok_c[0]: return (False, List[UInt8]())
            var HCLEN = 4 + ok_c[1]

            var ORDER = [16,17,18, 0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
            var cl_lengths = List[Int]()
            var ii = 0
            while ii < 19:
                cl_lengths.append(0)
                ii += 1
            ii = 0
            while ii < HCLEN:
                var okb = br.read_bits(3)
                if not okb[0]: return (False, List[UInt8]())
                cl_lengths[ORDER[ii]] = okb[1]
                ii += 1

            var ok_cl = _build_huffman(cl_lengths, 7)
            if not ok_cl[0]: return (False, List[UInt8]())
            var cl_table = ok_cl[1].copy()

            var ok_ll = _read_code_lengths(br, HLIT, cl_table)
            if not ok_ll[0]: return (False, List[UInt8]())
            var litlen_len = ok_ll[1].copy()

            var ok_dl = _read_code_lengths(br, HDIST, cl_table)
            if not ok_dl[0]: return (False, List[UInt8]())
            var dist_len = ok_dl[1].copy()

            var ok_lit  = _build_huffman(litlen_len, 15)
            if not ok_lit[0]: return (False, List[UInt8]())
            var ok_dist = _build_huffman(dist_len, 15)
            if not ok_dist[0]: return (False, List[UInt8]())

            if not _inflate_block_with_tables(br, ok_lit[1], ok_dist[1], out):
                return (False, List[UInt8]())

        else:
            return (False, List[UInt8]())

        if btype == 0 or btype == 1 or btype == 2:
            # continue; the spec allows multiple blocks
            pass

        if bfinal == 1:
            break

    return (True, out.copy())
