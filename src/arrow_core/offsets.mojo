# Offsets buffer for variable-size arrays (e.g., strings).

struct Offsets:
    data: List[Int]

    fn __init__(inout self):
        self.data = List[Int]()
        self.data.append(0)

    fn append_offset(inout self, o: Int):
        self.data.append(o)

    fn len(self) -> Int:
        return self.data.len()
