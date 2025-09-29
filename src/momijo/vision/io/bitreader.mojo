# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: momijo/vision/io/bitreader.mojo
# Description: MSB-first JPEG bitstream reader with 0xFF/0x00 stuffing handling.

struct BitReader:
    var ptr: UnsafePointer[UInt8]  # non-owning input buffer
    var length: Int                # total bytes available
    var pos: Int                   # current byte offset
    var bit_buf: UInt32            # MSB-first shift register
    var bit_count: Int             # number of valid bits in bit_buf
    var eof: Bool                  # set when hitting non-stuffed 0xFF or end

    fn __init__(out self, ptr: UnsafePointer[UInt8], length: Int):
        self.ptr = ptr
        self.length = length
        self.pos = 0
        self.bit_buf = 0
        self.bit_count = 0
        self.eof = False

    # Pull whole bytes until at least `need` bits are available (or EOF).
    # Handles JPEG byte stuffing: 0xFF 0x00 -> literal 0xFF data byte.
    # If we see 0xFF followed by a non-zero byte, we stop (marker encountered).
    fn _fill(mut self, need: Int):
        while self.bit_count < need and self.pos < self.length and not self.eof:
            var b = self.ptr[self.pos]
            self.pos = self.pos + 1

            if b == UInt8(0xFF):
                if self.pos < self.length:
                    var nextb = self.ptr[self.pos]
                    if nextb == UInt8(0x00):
                        # stuffed 0xFF -> skip the 0x00 and treat 0xFF as data
                        self.pos = self.pos + 1
                    else:
                        # marker boundary (0xFF followed by non-zero)
                        self.eof = True
                        break
                else:
                    # 0xFF at the very end -> treat as EOF
                    self.eof = True
                    break

            # push byte into MSB-first buffer
            self.bit_buf = (self.bit_buf << UInt32(8)) | UInt32(b)
            self.bit_count = self.bit_count + 8

    # Read n bits (1..24), MSB-first. Returns 0 on underflow/EOF.
    fn read_bits(mut self, n: Int) -> UInt32:
        if n <= 0:
            return 0
        if n > 24:
            self._fill(24)
        else:
            self._fill(n)

        if self.bit_count < n:
            # underflow or EOF
            self.eof = True
            return 0

        var shift = UInt32(self.bit_count - n)
        var mask = (UInt32(1) << UInt32(n)) - UInt32(1)
        var outv = (self.bit_buf >> shift) & mask

        # consume n bits
        self.bit_buf = self.bit_buf & ((UInt32(1) << shift) - UInt32(1))
        self.bit_count = self.bit_count - n
        return outv

    # Peek n bits without consuming. Returns 0 on underflow/EOF.
    fn peek_bits(mut self, n: Int) -> UInt32:
        if n <= 0:
            return 0
        if n > 24:
            self._fill(24)
        else:
            self._fill(n)

        if self.bit_count < n:
            self.eof = True
            return 0

        var shift = UInt32(self.bit_count - n)
        var mask = (UInt32(1) << UInt32(n)) - UInt32(1)
        return (self.bit_buf >> shift) & mask

    # Skip n bits (best-effort; may set eof if not enough data).
    fn skip_bits(mut self, n: Int):
        if n <= 0:
            return
        self._fill(n)
        if self.bit_count < n:
            self.bit_count = 0
            self.bit_buf = 0
            self.eof = True
            return
        var shift = UInt32(self.bit_count - n)
        self.bit_buf = self.bit_buf & ((UInt32(1) << shift) - UInt32(1))
        self.bit_count = self.bit_count - n

    # Discard residual bits to next byte boundary.
    fn align_byte(mut self):
        var rem = self.bit_count % 8
        if rem != 0:
            _ = self.read_bits(rem)

    # Number of source bytes consumed (moved into bit_buf).
    fn bytes_consumed(self) -> Int:
        return self.pos
