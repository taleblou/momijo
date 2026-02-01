# MIT License
# Baseline JPEG encoder (Mojo) — append-based lists, integer ops, no mut at call sites.

from collections.list import List
from momijo.vision.io.bitwriter import BitWriter

@always_inline
fn _dbg(msg: String):
    print(msg)


# Optional wrappers: if BitWriter has start_entropy/finish_entropy, call them; otherwise no-op.
@always_inline
fn _try_start_entropy(mut bw: BitWriter):
    # Some Mojo versions may not have these; calls are no-op if missing at compile time.
    # We'll just ignore errors by not referencing non-existent methods.
    _ = bw.byte_align()  # keep alignment before starting entropy

@always_inline
fn _try_finish_entropy(mut bw: BitWriter):
    _ = bw.byte_align()


@always_inline
fn write_u16_be(mut bw: BitWriter, v: Int) -> Bool:
    if not bw.write_u8(UInt8((v >> 8) & 0xFF)): return False
    if not bw.write_u8(UInt8(v & 0xFF)): return False
    return True

@always_inline
fn _clamp_u8(x: Int) -> UInt8:
    var y = x
    if y < 0: y = 0
    if y > 255: y = 255
    return UInt8(y)

fn _base_ql() -> List[Int]:
    var L = List[Int]()
    L.append(16)
    L.append(11)
    L.append(10)
    L.append(16)
    L.append(24)
    L.append(40)
    L.append(51)
    L.append(61)
    L.append(12)
    L.append(12)
    L.append(14)
    L.append(19)
    L.append(26)
    L.append(58)
    L.append(60)
    L.append(55)
    L.append(14)
    L.append(13)
    L.append(16)
    L.append(24)
    L.append(40)
    L.append(57)
    L.append(69)
    L.append(56)
    L.append(14)
    L.append(17)
    L.append(22)
    L.append(29)
    L.append(51)
    L.append(87)
    L.append(80)
    L.append(62)
    L.append(18)
    L.append(22)
    L.append(37)
    L.append(56)
    L.append(68)
    L.append(109)
    L.append(103)
    L.append(77)
    L.append(24)
    L.append(35)
    L.append(55)
    L.append(64)
    L.append(81)
    L.append(104)
    L.append(113)
    L.append(92)
    L.append(49)
    L.append(64)
    L.append(78)
    L.append(87)
    L.append(103)
    L.append(121)
    L.append(120)
    L.append(101)
    L.append(72)
    L.append(92)
    L.append(95)
    L.append(98)
    L.append(112)
    L.append(100)
    L.append(103)
    L.append(99)
    return L.copy()

fn _base_qc() -> List[Int]:
    var L = List[Int]()
    L.append(17)
    L.append(18)
    L.append(24)
    L.append(47)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(18)
    L.append(21)
    L.append(26)
    L.append(66)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(24)
    L.append(26)
    L.append(56)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(47)
    L.append(66)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    L.append(99)
    return L.copy()

fn _zz_order() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(1)
    L.append(5)
    L.append(6)
    L.append(14)
    L.append(15)
    L.append(27)
    L.append(28)
    L.append(2)
    L.append(4)
    L.append(7)
    L.append(13)
    L.append(16)
    L.append(26)
    L.append(29)
    L.append(42)
    L.append(3)
    L.append(8)
    L.append(12)
    L.append(17)
    L.append(25)
    L.append(30)
    L.append(41)
    L.append(43)
    L.append(9)
    L.append(11)
    L.append(18)
    L.append(24)
    L.append(31)
    L.append(40)
    L.append(44)
    L.append(53)
    L.append(10)
    L.append(19)
    L.append(23)
    L.append(32)
    L.append(39)
    L.append(45)
    L.append(52)
    L.append(54)
    L.append(20)
    L.append(22)
    L.append(33)
    L.append(38)
    L.append(46)
    L.append(51)
    L.append(55)
    L.append(60)
    L.append(21)
    L.append(34)
    L.append(37)
    L.append(47)
    L.append(50)
    L.append(56)
    L.append(59)
    L.append(61)
    L.append(35)
    L.append(36)
    L.append(48)
    L.append(49)
    L.append(57)
    L.append(58)
    L.append(62)
    L.append(63)
    return L.copy()

fn _bits_dc_y() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(1)
    L.append(5)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(0)
    L.append(0)
    L.append(0)
    L.append(0)
    L.append(0)
    return L.copy()

fn _vals_dc_y() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(1)
    L.append(2)
    L.append(3)
    L.append(4)
    L.append(5)
    L.append(6)
    L.append(7)
    L.append(8)
    L.append(9)
    L.append(10)
    L.append(11)
    return L.copy()

fn _bits_ac_y() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(2)
    L.append(1)
    L.append(3)
    L.append(3)
    L.append(2)
    L.append(4)
    L.append(3)
    L.append(5)
    L.append(5)
    L.append(4)
    L.append(4)
    L.append(0)
    L.append(0)
    L.append(1)
    L.append(125)
    return L.copy()

fn _vals_ac_y() -> List[Int]:
    var L = List[Int]()
    L.append(1)
    L.append(2)
    L.append(3)
    L.append(0)
    L.append(4)
    L.append(17)
    L.append(5)
    L.append(18)
    L.append(33)
    L.append(49)
    L.append(65)
    L.append(6)
    L.append(19)
    L.append(81)
    L.append(97)
    L.append(7)
    L.append(34)
    L.append(113)
    L.append(20)
    L.append(50)
    L.append(129)
    L.append(145)
    L.append(161)
    L.append(8)
    L.append(35)
    L.append(66)
    L.append(177)
    L.append(193)
    L.append(21)
    L.append(82)
    L.append(209)
    L.append(240)
    L.append(36)
    L.append(51)
    L.append(98)
    L.append(114)
    L.append(130)
    L.append(9)
    L.append(10)
    L.append(22)
    L.append(23)
    L.append(24)
    L.append(25)
    L.append(26)
    L.append(37)
    L.append(38)
    L.append(39)
    L.append(40)
    L.append(41)
    L.append(42)
    L.append(52)
    L.append(53)
    L.append(54)
    L.append(55)
    L.append(56)
    L.append(57)
    L.append(58)
    L.append(67)
    L.append(68)
    L.append(69)
    L.append(70)
    L.append(71)
    L.append(72)
    L.append(73)
    L.append(74)
    L.append(83)
    L.append(84)
    L.append(85)
    L.append(86)
    L.append(87)
    L.append(88)
    L.append(89)
    L.append(90)
    L.append(99)
    L.append(100)
    L.append(101)
    L.append(102)
    L.append(103)
    L.append(104)
    L.append(105)
    L.append(106)
    L.append(115)
    L.append(116)
    L.append(117)
    L.append(118)
    L.append(119)
    L.append(120)
    L.append(121)
    L.append(122)
    L.append(131)
    L.append(132)
    L.append(133)
    L.append(134)
    L.append(135)
    L.append(136)
    L.append(137)
    L.append(138)
    L.append(146)
    L.append(147)
    L.append(148)
    L.append(149)
    L.append(150)
    L.append(151)
    L.append(152)
    L.append(153)
    L.append(154)
    L.append(162)
    L.append(163)
    L.append(164)
    L.append(165)
    L.append(166)
    L.append(167)
    L.append(168)
    L.append(169)
    L.append(170)
    L.append(178)
    L.append(179)
    L.append(180)
    L.append(181)
    L.append(182)
    L.append(183)
    L.append(184)
    L.append(185)
    L.append(186)
    L.append(194)
    L.append(195)
    L.append(196)
    L.append(197)
    L.append(198)
    L.append(199)
    L.append(200)
    L.append(201)
    L.append(202)
    L.append(210)
    L.append(211)
    L.append(212)
    L.append(213)
    L.append(214)
    L.append(215)
    L.append(216)
    L.append(217)
    L.append(218)
    L.append(225)
    L.append(226)
    L.append(227)
    L.append(228)
    L.append(229)
    L.append(230)
    L.append(231)
    L.append(232)
    L.append(233)
    L.append(234)
    L.append(241)
    L.append(242)
    L.append(243)
    L.append(244)
    L.append(245)
    L.append(246)
    L.append(247)
    L.append(248)
    L.append(249)
    L.append(250)
    return L.copy()

fn _bits_dc_c() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(3)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(1)
    L.append(0)
    L.append(0)
    L.append(0)
    L.append(0)
    L.append(0)
    return L.copy()

fn _vals_dc_c() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(1)
    L.append(2)
    L.append(3)
    L.append(4)
    L.append(5)
    L.append(6)
    L.append(7)
    L.append(8)
    L.append(9)
    L.append(10)
    L.append(11)
    return L.copy()

fn _bits_ac_c() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(2)
    L.append(1)
    L.append(2)
    L.append(4)
    L.append(4)
    L.append(3)
    L.append(4)
    L.append(7)
    L.append(5)
    L.append(4)
    L.append(4)
    L.append(0)
    L.append(1)
    L.append(2)
    L.append(119)
    return L.copy()

fn _vals_ac_c() -> List[Int]:
    var L = List[Int]()
    L.append(0)
    L.append(1)
    L.append(2)
    L.append(3)
    L.append(17)
    L.append(4)
    L.append(5)
    L.append(33)
    L.append(49)
    L.append(6)
    L.append(18)
    L.append(65)
    L.append(81)
    L.append(7)
    L.append(97)
    L.append(113)
    L.append(19)
    L.append(34)
    L.append(50)
    L.append(129)
    L.append(8)
    L.append(20)
    L.append(66)
    L.append(145)
    L.append(161)
    L.append(177)
    L.append(193)
    L.append(9)
    L.append(35)
    L.append(51)
    L.append(82)
    L.append(240)
    L.append(21)
    L.append(98)
    L.append(114)
    L.append(209)
    L.append(10)
    L.append(22)
    L.append(36)
    L.append(52)
    L.append(225)
    L.append(37)
    L.append(241)
    L.append(23)
    L.append(24)
    L.append(25)
    L.append(26)
    L.append(38)
    L.append(39)
    L.append(40)
    L.append(41)
    L.append(42)
    L.append(53)
    L.append(54)
    L.append(55)
    L.append(56)
    L.append(57)
    L.append(58)
    L.append(67)
    L.append(68)
    L.append(69)
    L.append(70)
    L.append(71)
    L.append(72)
    L.append(73)
    L.append(74)
    L.append(83)
    L.append(84)
    L.append(85)
    L.append(86)
    L.append(87)
    L.append(88)
    L.append(89)
    L.append(90)
    L.append(99)
    L.append(100)
    L.append(101)
    L.append(102)
    L.append(103)
    L.append(104)
    L.append(105)
    L.append(106)
    L.append(115)
    L.append(116)
    L.append(117)
    L.append(118)
    L.append(119)
    L.append(120)
    L.append(121)
    L.append(122)
    L.append(130)
    L.append(131)
    L.append(132)
    L.append(133)
    L.append(134)
    L.append(135)
    L.append(136)
    L.append(137)
    L.append(138)
    L.append(146)
    L.append(147)
    L.append(148)
    L.append(149)
    L.append(150)
    L.append(151)
    L.append(152)
    L.append(153)
    L.append(154)
    L.append(162)
    L.append(163)
    L.append(164)
    L.append(165)
    L.append(166)
    L.append(167)
    L.append(168)
    L.append(169)
    L.append(170)
    L.append(178)
    L.append(179)
    L.append(180)
    L.append(181)
    L.append(182)
    L.append(183)
    L.append(184)
    L.append(185)
    L.append(186)
    L.append(194)
    L.append(195)
    L.append(196)
    L.append(197)
    L.append(198)
    L.append(199)
    L.append(200)
    L.append(201)
    L.append(202)
    L.append(210)
    L.append(211)
    L.append(212)
    L.append(213)
    L.append(214)
    L.append(215)
    L.append(216)
    L.append(217)
    L.append(218)
    L.append(226)
    L.append(227)
    L.append(228)
    L.append(229)
    L.append(230)
    L.append(231)
    L.append(232)
    L.append(233)
    L.append(234)
    L.append(242)
    L.append(243)
    L.append(244)
    L.append(245)
    L.append(246)
    L.append(247)
    L.append(248)
    L.append(249)
    L.append(250)
    return L.copy()

@always_inline
fn _write_marker(mut bw: BitWriter, m: UInt8) -> Bool:
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(m): return False
    return True

@always_inline
fn write_soi(mut bw: BitWriter) -> Bool:
    return _write_marker(bw, 0xD8)

@always_inline
fn write_eoi(mut bw: BitWriter) -> Bool:
    _ = bw.byte_align()
    return _write_marker(bw, 0xD9)

@always_inline
fn _write_app0_jfif(mut bw: BitWriter) -> Bool:
    if not _write_marker(bw, 0xE0): return False
    if not bw.write_u8(0x00): return False
    if not bw.write_u8(0x10): return False  # length 16
    var sig = List[UInt8]()
    sig.append(0x4A); sig.append(0x46); sig.append(0x49); sig.append(0x46); sig.append(0x00)
    sig.append(0x01); sig.append(0x01); sig.append(0x00)
    sig.append(0x00); sig.append(0x01); sig.append(0x00); sig.append(0x01)
    sig.append(0x00); sig.append(0x00)
    var i = 0
    while i < len(sig):
        if not bw.write_u8(sig[i]): return False
        i += 1
    return True

@always_inline
fn _scale_q(v: Int, S: Int) -> UInt8:
    var x = (v * S + 50) // 100
    if x < 1:   x = 1
    if x > 255: x = 255
    return UInt8(x)

@always_inline
fn _quality_scale(quality: Int) -> Int:
    var q = quality
    if q < 1: q = 1
    if q > 100: q = 100
    if q < 50:
        return 5000 // q
    return 200 - 2 * q

@always_inline
fn _scale_q_val(v: Int, S: Int) -> Int:
    var x: Int = (v * S + 50) // 100
    if x < 1:   x = 1
    if x > 255: x = 255
    return x

@always_inline
fn _write_dqt_luma_chroma_scaled(mut bw: BitWriter, quality: Int) -> Bool:
    var S = _quality_scale(quality)
    var QL = _base_ql()
    var QC = _base_qc()
    var ZZ = _zz_order()

    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xDB): return False
    if not bw.write_u8(0x00): return False
    if not bw.write_u8(0x84): return False

    if not bw.write_u8(0x00): return False
    var i = 0
    while i < 64:
        var idx = ZZ[i]
        if not bw.write_u8(_scale_q(QL[idx], S)): return False
        i += 1

    if not bw.write_u8(0x01): return False
    i = 0
    while i < 64:
        var idx2 = ZZ[i]
        if not bw.write_u8(_scale_q(QC[idx2], S)): return False
        i += 1
    return True

@always_inline
fn _write_sof0(mut bw: BitWriter, width: Int, height: Int, channels: Int) -> Bool:
    if width <= 0 or height <= 0: return False
    if not (channels == 1 or channels == 3): return False

    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xC0): return False
    var Nf = channels
    var Lf = 8 + 3 * Nf
    if not write_u16_be(bw, Lf): return False
    if not bw.write_u8(0x08): return False
    if not write_u16_be(bw, height): return False
    if not write_u16_be(bw, width):  return False
    if not bw.write_u8(UInt8(Nf)): return False

    if channels == 1:
        if not bw.write_u8(0x01): return False
        if not bw.write_u8(0x11): return False
        if not bw.write_u8(0x00): return False
        return True

    if not bw.write_u8(0x01): return False
    if not bw.write_u8(0x11): return False
    if not bw.write_u8(0x00): return False
    if not bw.write_u8(0x02): return False
    if not bw.write_u8(0x11): return False
    if not bw.write_u8(0x01): return False
    if not bw.write_u8(0x03): return False
    if not bw.write_u8(0x11): return False
    if not bw.write_u8(0x01): return False
    return True

fn _write_std_huffman_tables(mut bw: BitWriter) -> Bool:
    # DC Luma
    var bits_dc_y = _bits_dc_y()
    var val_dc_y  = _vals_dc_y()
    var n = 0
    var i = 0
    while i < 16:
        n += bits_dc_y[i]
        i += 1
    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xC4): return False
    if not write_u16_be(bw, 3 + 16 + n): return False
    if not bw.write_u8(0x00): return False
    i = 0
    while i < 16:
        if not bw.write_u8(UInt8(bits_dc_y[i])): return False
        i += 1
    i = 0
    while i < n:
        if not bw.write_u8(UInt8(val_dc_y[i])): return False
        i += 1

    # AC Luma
    var bits_ac_y = _bits_ac_y()
    var val_ac_y  = _vals_ac_y()
    n = 0
    i = 0
    while i < 16:
        n += bits_ac_y[i]
        i += 1
    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xC4): return False
    if not write_u16_be(bw, 3 + 16 + n): return False
    if not bw.write_u8(0x10): return False
    i = 0
    while i < 16:
        if not bw.write_u8(UInt8(bits_ac_y[i])): return False
        i += 1
    i = 0
    while i < n:
        if not bw.write_u8(UInt8(val_ac_y[i])): return False
        i += 1

    # DC Chroma
    var bits_dc_c = _bits_dc_c()
    var val_dc_c  = _vals_dc_c()
    n = 0
    i = 0
    while i < 16:
        n += bits_dc_c[i]
        i += 1
    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xC4): return False
    if not write_u16_be(bw, 3 + 16 + n): return False
    if not bw.write_u8(0x01): return False
    i = 0
    while i < 16:
        if not bw.write_u8(UInt8(bits_dc_c[i])): return False
        i += 1
    i = 0
    while i < n:
        if not bw.write_u8(UInt8(val_dc_c[i])): return False
        i += 1

    # AC Chroma
    var bits_ac_c = _bits_ac_c()
    var val_ac_c  = _vals_ac_c()
    n = 0
    i = 0
    while i < 16:
        n += bits_ac_c[i]
        i += 1
    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xC4): return False
    if not write_u16_be(bw, 3 + 16 + n): return False
    if not bw.write_u8(0x11): return False
    i = 0
    while i < 16:
        if not bw.write_u8(UInt8(bits_ac_c[i])): return False
        i += 1
    i = 0
    while i < n:
        if not bw.write_u8(UInt8(val_ac_c[i])): return False
        i += 1
    return True

@always_inline
fn _write_sos(mut bw: BitWriter, channels: Int) -> Bool:
    if not (channels == 1 or channels == 3): return False
    _ = bw.byte_align()
    if not bw.write_u8(0xFF): return False
    if not bw.write_u8(0xDA): return False
    var Ns = channels
    var Ls = 6 + 2 * Ns
    if not write_u16_be(bw, Ls): return False
    if not bw.write_u8(UInt8(Ns)): return False
    if channels == 1:
        if not bw.write_u8(0x01): return False
        if not bw.write_u8(0x00): return False
    else:
        if not bw.write_u8(0x01): return False
        if not bw.write_u8(0x00): return False
        if not bw.write_u8(0x02): return False
        if not bw.write_u8(0x11): return False
        if not bw.write_u8(0x03): return False
        if not bw.write_u8(0x11): return False
    if not bw.write_u8(0x00): return False
    if not bw.write_u8(0x3F): return False
    if not bw.write_u8(0x00): return False
    return True

fn _build_huff_tables(bits: List[Int], vals: List[Int],
                      mut out_codes: List[Int], mut out_sizes: List[Int]) -> Bool:
    if len(bits) != 16: return False
    var sizes = List[Int]()
    var i = 0
    while i < 16:
        var cnt = bits[i]
        var j = 0
        while j < cnt:
            sizes.append(i + 1)
            j += 1
        i += 1
    var n = len(vals)
    if len(sizes) < n: return False
    var codes = List[Int]()
    codes.resize(n, 0)
    var code = 0
    var k = 0
    i = 0
    while i < 16:
        var cntL = bits[i]
        var j2 = 0
        while j2 < cntL and k < n:
            codes[k] = code
            k += 1
            code += 1
            j2 += 1
        code = code << 1
        i += 1
    if len(out_codes) < 256: out_codes.resize(256, 0)
    if len(out_sizes) < 256: out_sizes.resize(256, 0)
    var t = 0
    while t < n:
        var sym = vals[t]
        if sym < 0 or sym > 255: return False
        out_codes[sym] = codes[t]
        out_sizes[sym] = sizes[t]
        t += 1
    return True

@always_inline
fn _emit_huff(mut bw: BitWriter, code: Int, size: Int) -> Bool:
    if not bw.write_bits(code, size): return False
    return True

@always_inline
fn _dc_category(v: Int) -> Int:
    var a = v
    if a < 0: a = -a
    var cat = 0
    while a != 0:
        cat += 1
        a >>= 1
    return cat

@always_inline
fn _dc_bits(v: Int, k: Int) -> Int:
    if k == 0: return 0
    if v >= 0: return v
    var mag = -v
    var mask = (1 << k) - 1
    return mag ^ mask

fn _encode_image_entropy(
    mut bw: BitWriter,
    ptr: UnsafePointer[UInt8],
    width: Int,
    height: Int,
    channels: Int,
    quality: Int = 90
) -> Bool:
    if not (channels == 1 or channels == 3):
        return False

    var dc_y_codes = List[Int](); var dc_y_sizes = List[Int]()
    var ac_y_codes = List[Int](); var ac_y_sizes = List[Int]()
    var dc_c_codes = List[Int](); var dc_c_sizes = List[Int]()
    var ac_c_codes = List[Int](); var ac_c_sizes = List[Int]()

    if not _build_huff_tables(_bits_dc_y(), _vals_dc_y(), dc_y_codes, dc_y_sizes):
        return False
    if not _build_huff_tables(_bits_ac_y(), _vals_ac_y(), ac_y_codes, ac_y_sizes):
        return False
    if not _build_huff_tables(_bits_dc_c(), _vals_dc_c(), dc_c_codes, dc_c_sizes):
        return False
    if not _build_huff_tables(_bits_ac_c(), _vals_ac_c(), ac_c_codes, ac_c_sizes):
        return False

    var S = _quality_scale(quality)
    var QL = _base_ql()
    var QC = _base_qc()
    var qY0: Int = _scale_q_val(QL[0], S)
    if qY0 <= 0:
        qY0 = 1
    var qC0: Int = _scale_q_val(QC[0], S)
    if qC0 <= 0:
        qC0 = 1

    var predY: Int = 0
    var predCb: Int = 0
    var predCr: Int = 0

    var by = 0
    while by < height:
        var bx = 0
        while bx < width:
            var sumY: Float64 = 0.0
            var y0 = by
            while y0 < by + 8 and y0 < height:
                var x0 = bx
                while x0 < bx + 8 and x0 < width:
                    if channels == 1:
                        var Yv = ptr[(y0 * width + x0) * 1 + 0]
                        sumY += Float64(Int(Yv) - 128)
                    else:
                        var r = ptr[(y0 * width + x0) * 3 + 0]
                        var g = ptr[(y0 * width + x0) * 3 + 1]
                        var b = ptr[(y0 * width + x0) * 3 + 2]
                        var Yr = 0.299 * Float64(Int(r)) + 0.587 * Float64(Int(g)) + 0.114 * Float64(Int(b))
                        sumY += (Yr - 128.0)
                    x0 += 1
                y0 += 1

            var dcY: Int = Int(sumY)
            var qdcY: Int = dcY // qY0
            var diffY: Int = qdcY - predY
            var catY: Int = _dc_category(diffY)
            var codeY = dc_y_codes[catY]
            var sizeY = dc_y_sizes[catY]
            if sizeY == 0:
                return False
            if not _emit_huff(bw, codeY, sizeY):
                return False
            if catY > 0:
                var bitsY = _dc_bits(diffY, catY)
                if not bw.write_bits(bitsY, catY):
                    return False
            predY = qdcY

            var eob_code = ac_y_codes[0x00]
            var eob_size = ac_y_sizes[0x00]
            if eob_size == 0:
                return False
            if not _emit_huff(bw, eob_code, eob_size):
                return False

            if channels == 3:
                var sumCb: Float64 = 0.0
                var sumCr: Float64 = 0.0
                y0 = by
                while y0 < by + 8 and y0 < height:
                    var x1 = bx
                    while x1 < bx + 8 and x1 < width:
                        var r2 = ptr[(y0 * width + x1) * 3 + 0]
                        var g2 = ptr[(y0 * width + x1) * 3 + 1]
                        var b2 = ptr[(y0 * width + x1) * 3 + 2]
                        var Rf = Float64(Int(r2))
                        var Gf = Float64(Int(g2))
                        var Bf = Float64(Int(b2))
                        var Cbf = -0.168736 * Rf - 0.331264 * Gf + 0.5 * Bf + 128.0
                        var Crf =  0.5 * Rf - 0.418688 * Gf - 0.081312 * Bf + 128.0
                        sumCb += (Cbf - 128.0)
                        sumCr += (Crf - 128.0)
                        x1 += 1
                    y0 += 1

                var dcCb: Int = Int(sumCb)
                var dcCr: Int = Int(sumCr)
                var qdcCb: Int = dcCb // qC0
                var qdcCr: Int = dcCr // qC0

                var diffCb: Int = qdcCb - predCb
                var catCb: Int = _dc_category(diffCb)
                var codeCb = dc_c_codes[catCb]
                var sizeCb = dc_c_sizes[catCb]
                if sizeCb == 0:
                    return False
                if not _emit_huff(bw, codeCb, sizeCb):
                    return False
                if catCb > 0:
                    var bitsCb = _dc_bits(diffCb, catCb)
                    if not bw.write_bits(bitsCb, catCb):
                        return False
                predCb = qdcCb

                var eobCb_code = ac_c_codes[0x00]
                var eobCb_size = ac_c_sizes[0x00]
                if eobCb_size == 0:
                    return False
                if not _emit_huff(bw, eobCb_code, eobCb_size):
                    return False

                var diffCr: Int = qdcCr - predCr
                var catCr: Int = _dc_category(diffCr)
                var codeCr = dc_c_codes[catCr]
                var sizeCr = dc_c_sizes[catCr]
                if sizeCr == 0:
                    return False
                if not _emit_huff(bw, codeCr, sizeCr):
                    return False
                if catCr > 0:
                    var bitsCr = _dc_bits(diffCr, catCr)
                    if not bw.write_bits(bitsCr, catCr):
                        return False
                predCr = qdcCr

                var eobCr_code = ac_c_codes[0x00]
                var eobCr_size = ac_c_sizes[0x00]
                if eobCr_size == 0:
                    return False
                if not _emit_huff(bw, eobCr_code, eobCr_size):
                    return False

            bx += 8
        by += 8

    if not bw.byte_align():
        return False
    return True


fn encode_jpeg(
    ptr: UnsafePointer[UInt8],
    width: Int,
    height: Int,
    out_buf: UnsafePointer[UInt8],
    out_max: Int,
    channels: Int = 3,
    quality: Int = 90
) -> (Bool, Int):
    if width <= 0 or height <= 0: return (False, 0)
    if not (channels == 1 or channels == 3): return (False, 0)

    var bw = BitWriter(out_buf, out_max)
    _dbg("STAGE: SOI")
    if not write_soi(bw):
        _dbg("FAIL: write_soi")
        return (False, 0)
    _dbg("STAGE: APP0")
    if not _write_app0_jfif(bw):
        _dbg("FAIL: app0")
        return (False, 0)
        _dbg("STAGE: DQT")
    if not _write_dqt_luma_chroma_scaled(bw, quality):
        _dbg("FAIL: dqt")
        return (False, 0)
        _dbg("STAGE: SOF0")
    if not _write_sof0(bw, width, height, channels):
        _dbg("FAIL: sof0")
        return (False, 0)
        _dbg("STAGE: DHT")
    if not _write_std_huffman_tables(bw):
        _dbg("FAIL: dht")
        return (False, 0)
        _try_start_entropy(bw)
        _dbg("STAGE: SOS")
    if not _write_sos(bw, channels):
        _dbg("FAIL: sos")
        return (False, 0)

        _dbg("STAGE: ENTROPY")
    if not _encode_image_entropy(bw, ptr, width, height, channels, quality):
            return (False, 0)

        _try_finish_entropy(bw)

        _dbg("STAGE: EOI")
    if not write_eoi(bw):
        _dbg("FAIL: eoi")
        return (False, 0)
    if not bw.good(): return (False, 0)
    var n = bw.tell()
    return (True, n)
