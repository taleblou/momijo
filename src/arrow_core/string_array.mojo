# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid

struct StringArray(Copyable, Movable, Sized):
    var data: List[String]
    var validity: Bitmap

    fn __init__(out self):
        self.data = List[String]()
        self.validity = Bitmap(0, True)

    fn __len__(self) -> Int:
        return len(self.data)

    fn len(self) -> Int:
        return len(self.data)

    fn push(mut self, s: String, valid: Bool = True):
        self.data.append(s)
        var n = len(self.data)
        self.validity = Bitmap(n, True)
        if not valid:
            bitmap_set_valid(self.validity, n - 1, False)

    fn get(self, i: Int) -> String:
        return self.data[i]
