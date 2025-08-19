# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


struct Bitmap(Copyable, Movable, EqualityComparable, Sized):
    var bytes: List[UInt8]
    var nbits: Int

    fn __init__(out self, nbits: Int, all_valid: Bool = True):
        self.nbits = nbits
        var nbytes = (nbits + 7) // 8
        self.bytes = List[UInt8]()
        var fill: UInt8 = UInt8(255) if all_valid else UInt8(0)
        var i = 0
        while i < nbytes:
            self.bytes.append(fill)
            i += 1

    fn __copyinit__(out self, other: Bitmap):
        self.nbits = other.nbits
        self.bytes = List[UInt8]()
        var i = 0
        while i < len(other.bytes):
            self.bytes.append(other.bytes[i])
            i += 1

    fn __len__(self) -> Int:
        return self.nbits

    fn __eq__(self, other: Bitmap) -> Bool:
        if self.nbits != other.nbits: return False
        if len(self.bytes) != len(other.bytes): return False
        var i = 0
        while i < len(self.bytes):
            if self.bytes[i] != other.bytes[i]: return False
            i += 1
        return True

    fn __ne__(self, other: Bitmap) -> Bool:
        return not self.__eq__(other)

    fn set_valid(mut self, idx: Int, v: Bool):
        var byte_idx = idx // 8
        var bit = idx % 8
        var mask: UInt8 = UInt8(1) << UInt8(bit)
        if v:
            self.bytes[byte_idx] = self.bytes[byte_idx] | mask
        else:
            self.bytes[byte_idx] = self.bytes[byte_idx] & ~mask

    fn count_valid(self) -> Int:
        var total = 0
        var i = 0
        var nbytes = len(self.bytes)
        while i < nbytes:
            var b = self.bytes[i]
            var j = 0
            while j < 8:
                if (b >> UInt8(j)) & UInt8(1) == UInt8(1):
                    total += 1
                j += 1
            i += 1
        var extra = (nbytes * 8) - self.nbits
        if extra > 0 and nbytes > 0:
            var cleared = 0
            var k = 0
            while k < extra:
                var bitmask: UInt8 = UInt8(1) << UInt8(7 - k)
                if (self.bytes[nbytes - 1] & bitmask) == bitmask:
                    cleared += 1
                k += 1
            total -= cleared
        return total

fn bitmap_set_valid(mut b: Bitmap, idx: Int, v: Bool):
    b.set_valid(idx, v)

fn bitmap_and(a: Bitmap, b: Bitmap) -> Bitmap:
    var nbytes = len(a.bytes) if len(a.bytes) < len(b.bytes) else len(b.bytes)
    var out = Bitmap(a.nbits if a.nbits < b.nbits else b.nbits, False)
    while len(out.bytes) < nbytes:
        out.bytes.append(UInt8(0))
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1
    return out

fn bitmap_or(a: Bitmap, b: Bitmap) -> Bitmap:
    var nbytes = len(a.bytes) if len(a.bytes) < len(b.bytes) else len(b.bytes)
    var out = Bitmap(a.nbits if a.nbits > b.nbits else b.nbits, False)
    while len(out.bytes) < nbytes:
        out.bytes.append(UInt8(0))
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] | b.bytes[i]
        i += 1
    return out

fn bitmap_not(a: Bitmap) -> Bitmap:
    var out = Bitmap(a.nbits, False)
    while len(out.bytes) < len(a.bytes):
        out.bytes.append(UInt8(0))
    var i = 0
    while i < len(a.bytes):
        out.bytes[i] = ~a.bytes[i]
        i += 1
    return out
