# Bitmap implementation compatible with Mojo's current syntax (no 'export' statements).
struct Bitmap(Copyable, Movable):
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
        # mask off unused bits in the last byte if marking all valid
        var rem = nbits % 8
        if rem != 0:
            var mask: UInt8 = UInt8(0)
            var k = 0
            while k < rem:
                mask = mask | (UInt8(1) << k)
                k += 1
            if all_valid:
                self.bytes[nbytes - 1] = self.bytes[nbytes - 1] & mask

    fn is_valid(self, idx: Int) -> Bool:
        var byte_idx = idx // 8
        var bit = idx % 8
        return ((self.bytes[byte_idx] >> bit) & UInt8(1)) == UInt8(1)
var byte_idx = idx // 8
        var bit = idx % 8
        if v:
            self.bytes[byte_idx] = self.bytes[byte_idx] | (UInt8(1) << bit)
        else:
            self.bytes[byte_idx] = self.bytes[byte_idx] & ~(UInt8(1) << bit)

    fn count_valid(self) -> Int:
        var nbytes = (self.nbits + 7) // 8
        var total = 0
        var i = 0
        while i < nbytes:
            var b = self.bytes[i]
            var c = 0
            var k = 0
            while k < 8:
                c += Int((b >> k) & UInt8(1))
                k += 1
            total += c
            i += 1
        return total

fn bitmap_and(a: Bitmap, b: Bitmap) -> Bitmap:
    var nbits = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(nbits, False)
    var nbytes = (nbits + 7) // 8
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1
    return out

fn bitmap_or(a: Bitmap, b: Bitmap) -> Bitmap:
    var nbits = a.nbits if a.nbits > b.nbits else b.nbits
    var out = Bitmap(nbits, False)
    var nbytes = (nbits + 7) // 8
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] | b.bytes[i]
        i += 1
    return out

fn bitmap_not(a: Bitmap) -> Bitmap:
    var out = Bitmap(a.nbits, False)
    var nbytes = (a.nbits + 7) // 8
    var i = 0
    while i < nbytes:
        out.bytes[i] = ~a.bytes[i]
        i += 1
    # clear unused high bits in the last byte
    var rem = a.nbits % 8
    if rem != 0:
        var mask: UInt8 = UInt8(0)
        var k = 0
        while k < rem:
            mask = mask | (UInt8(1) << k)
            k += 1
        out.bytes[nbytes - 1] = out.bytes[nbytes - 1] & mask
    return out


fn bitmap_set_valid(inout b: Bitmap, idx: Int, v: Bool):
    var byte_idx = idx // 8
    var bit = idx % 8
    if v:
        b.bytes[byte_idx] = b.bytes[byte_idx] | (UInt8(1) << bit)
    else:
        b.bytes[byte_idx] = b.bytes[byte_idx] & ~(UInt8(1) << bit)
