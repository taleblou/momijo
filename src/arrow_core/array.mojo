# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid

struct Array[T: Copyable & Movable](Copyable, Movable, Sized):
    var values: List[T]
    var validity: Bitmap

    fn __init__(out self, n: Int = 0, all_valid: Bool = True):
        self.values = List[T]()
        self.validity = Bitmap(n, all_valid)

    fn __len__(self) -> Int:
        return len(self.values)

    fn len(self) -> Int:
        return len(self.values)

    fn push(mut self, v: T, valid: Bool = True):
        self.values.append(v)
        var n = len(self.values)
        self.validity = Bitmap(n, True)
        if not valid:
            bitmap_set_valid(self.validity, n - 1, False)

    fn get(self, i: Int) -> T:
        return self.values[i]
