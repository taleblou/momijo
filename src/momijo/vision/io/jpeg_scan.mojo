from momijo.vision.io.bitreader import BitReader
from momijo.vision.io.huffman import HuffmanTable, receive_extend
from momijo.vision.io.idct import idct_8x8

# Decode one 8x8 block: DC + 63 AC, de-quantize, then IDCT in-place.
fn decode_block(
    mut br: BitReader,
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
    quant: UnsafePointer[UInt8],
    prev_dc: Int,
    dst: UnsafePointer[Int]
) -> Int:
    # clear block
    var i = 0
    while i < 64:
        dst[i] = 0
        i += 1

    # ---- DC ----
    var t = dc_table.decode(br)
    if t < 0:
        return -1
    var diff = receive_extend(br, t)
    var dc = prev_dc + diff
    dst[0] = dc * Int(quant[0])

    # ---- AC ----
    var k = 1
    while k < 64:
        var rs = ac_table.decode(br)
        if rs < 0:
            return -1
        if rs == 0:
            break  # EOB

        var r = (rs >> 4)
        var s = (rs & 0x0F)
        k = k + r
        if k >= 64:
            return -2

        var ac_val = receive_extend(br, s)
        dst[k] = ac_val * Int(quant[k])
        k = k + 1

    # inverse DCT in place
    idct_8x8(dst)
    return dc
