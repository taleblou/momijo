# Bitmap implementation compatible with Mojo's current syntax (no 'export' statements).
struct Bitmap:
    var bytes: List[UInt8]
    var nbits: Int

# Constructor: __init__(out self, nbits: Int, all_valid: Bool = True)
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

# Function is_valid(self, idx: Int) -> Bool
    fn is_valid(self, idx: Int) -> Bool:
        var byte_idx = idx // 8
        var bit = idx % 8
        return ((self.bytes[byte_idx] >> bit) & UInt8(1)) == UInt8(1)

# Function count_valid(self) -> Int
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


# Function bitmap_set_valid(mut b: Bitmap, idx: Int, v: Bool)
fn bitmap_set_valid(mut b: Bitmap, idx: Int, v: Bool):
    var byte_idx = idx // 8
    var bit = idx % 8
    if v:
        b.bytes[byte_idx] = b.bytes[byte_idx] | (UInt8(1) << bit)
    else:
        b.bytes[byte_idx] = b.bytes[byte_idx] & ~(UInt8(1) << bit)


# Function bitmap_and(read a: Bitmap, read b: Bitmap, out result: Bitmap)
fn bitmap_and(read a: Bitmap, read b: Bitmap, out result: Bitmap):
    var nbits = a.nbits if a.nbits < b.nbits else b.nbits
    result = Bitmap(nbits, False)
    var nbytes = (nbits + 7) // 8
    var i = 0
    while i < nbytes:
        result.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1

# Function bitmap_or(read a: Bitmap, read b: Bitmap, out result: Bitmap)
fn bitmap_or(read a: Bitmap, read b: Bitmap, out result: Bitmap):
    var nbits = a.nbits if a.nbits > b.nbits else b.nbits
    result = Bitmap(nbits, False)
    var nbytes = (nbits + 7) // 8
    var i = 0
    while i < nbytes:
        result.bytes[i] = a.bytes[i] | b.bytes[i]
        i += 1

# Function bitmap_not(read a: Bitmap, out result: Bitmap)
fn bitmap_not(read a: Bitmap, out result: Bitmap):
    result = Bitmap(a.nbits, False)
    var nbytes = (a.nbits + 7) // 8
    var i = 0
    while i < nbytes:
        result.bytes[i] = ~a.bytes[i]
        i += 1
    var rem = a.nbits % 8
    if rem != 0:
        var mask: UInt8 = UInt8(0)
        var k = 0
        while k < rem:
            mask = mask | (UInt8(1) << k)
            k += 1
        result.bytes[nbytes - 1] = result.bytes[nbytes - 1] & mask