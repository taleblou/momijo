# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/bit_writer.mojo

struct BitWriter:
    var buf: UnsafePointer[UInt8]
    var cap: Int
    var pos: Int          # written bytes
    var bit_acc: Int      # pending bits accumulator (LSB-aligned)
    var bit_count: Int    # number of pending bits
    var ok: Bool

    fn __init__(out self, buf: UnsafePointer[UInt8], cap: Int):
        self.buf = buf
        self.cap = cap
        self.pos = 0
        self.bit_acc = 0
        self.bit_count = 0
        self.ok = True

    @always_inline
    fn _put_byte(mut self, b: UInt8) -> Bool:
        if self.pos >= self.cap:
            self.ok = False
            return False
        self.buf[self.pos] = b
        self.pos += 1
        return True

    fn write_u8(mut self, b: UInt8) -> Bool:
        # Only valid on byte boundary (or for markers), caller ensures alignment.
        return self._put_byte(b)

    fn write_bits(mut self, bits: Int, nbits: Int) -> Bool:
        # Append 'nbits' LSBs of 'bits' into bit_acc; MSB-first emission.
        if nbits <= 0 or nbits > 16:
            self.ok = False
            return False
        var val = bits & ((1 << nbits) - 1)
        self.bit_acc = (self.bit_acc << nbits) | val
        self.bit_count += nbits
        while self.bit_count >= 8:
            self.bit_count -= 8
            var outb = UInt8((self.bit_acc >> self.bit_count) & 0xFF)
            if not self._put_byte(outb): return False
            # JPEG byte-stuffing: after 0xFF write 0x00
            if outb == 0xFF:
                if not self._put_byte(UInt8(0x00)): return False
        return True

    fn byte_align(mut self) -> Bool:
        if self.bit_count == 0:
            return True
        var pad = 8 - self.bit_count
        return self.write_bits(0, pad)

    fn tell(self) -> Int:
        return self.pos

    fn good(self) -> Bool:
        return self.ok
