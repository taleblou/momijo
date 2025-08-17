from arrow_core.bitmap import Bitmap
from arrow_core.buffer import Buffer

struct Array[T]:
    values: Buffer[T]
    validity: Bitmap

    fn __init__(inout self, n: Int = 0, all_valid: Bool = True):
        self.values = Buffer[T]()
        self.validity = Bitmap(n, all_valid)
        var i = 0
        while i < n:
            # default-init (may be zero)
            self.values.push(T())
            i += 1

    fn len(self) -> Int:
        return self.values.len()

    fn push(inout self, v: T, valid: Bool = True):
        self.values.push(v)
        self.validity.ensure_size(self.len(), True)
        self.validity.set_valid(self.len() - 1, valid)

    fn get(self, i: Int) -> T:
        return self.values.get(i)
