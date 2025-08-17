# Validity bitmap (null tracking).

struct Bitmap:
    bytes: List[UInt8]
    nbits: Int

    fn __init__(inout self, nbits: Int, all_valid: Bool = True):
        self.nbits = nbits
        let nbytes = (nbits + 7) // 8
        self.bytes = List[UInt8]()
        self.bytes.reserve(nbytes)
        let fill: UInt8 = 0xFF if all_valid else 0x00
        for i in range(nbytes):
            self.bytes.append(fill)

    fn ensure_size(inout self, nbits: Int, default_valid: Bool = True):
        if nbits <= self.nbits: return
        let nbytes_old = (self.nbits + 7) // 8
        let nbytes_new = (nbits + 7) // 8
        let fill: UInt8 = 0xFF if default_valid else 0x00
        for _ in range(nbytes_new - nbytes_old):
            self.bytes.append(fill)
        self.nbits = nbits

    fn is_valid(self, idx: Int) -> Bool:
        let byte_idx = idx // 8
        let bit = idx % 8
        return ((self.bytes[byte_idx] >> bit) & 1) == 1

    fn set_valid(inout self, idx: Int, v: Bool):
        let byte_idx = idx // 8
        let bit = idx % 8
        var b = self.bytes[byte_idx]
        if v:
            b = b | (1 << bit)
        else:
            b = b & ~(1 << bit)
        self.bytes[byte_idx] = b

fn bitmap_and(a: Bitmap, b: Bitmap) -> Bitmap:
    let n = a.bytes.len() if a.bytes.len() < b.bytes.len() else b.bytes.len()
    var out = Bitmap(a.nbits if a.nbits < b.nbits else b.nbits, False)
    var i = 0
    while i < n:
        out.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1
    return out

fn bitmap_not(a: Bitmap) -> Bitmap:
    var out = Bitmap(a.nbits, False)
    var i = 0
    while i < a.bytes.len():
        out.bytes[i] = ~a.bytes[i]
        i += 1
    return out
