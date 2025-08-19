
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid
from momijo.arrow_core.offsets import Offsets

struct ByteStringArray:
    data: List[UInt8]
    offsets: Offsets
    validity: Bitmap

    fn __init__(out self):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.validity = Bitmap(0, True)

    fn push_bytes(mut self, bytes: List[UInt8], valid: Bool = True):
        var i = 0
        while i < len(bytes):
            self.data.append(bytes[i])
            i += 1
        self.offsets.add_length(len(bytes))
        var n = self.len()
        self.validity = Bitmap(n, True)
        if not valid:
            bitmap_set_valid(self.validity, n - 1, False)

    fn len(self) -> Int:
        return len(self.offsets.data) - 1

    fn get_bytes(self, i: Int) -> List[UInt8]:
        var out = List[UInt8]()
        var start = self.offsets.data[i]
        var end = self.offsets.data[i + 1]
        var j = start
        while j < end:
            out.append(self.data[j])
            j += 1
        return out
