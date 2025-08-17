# Simple contiguous buffer abstraction.

struct Buffer[T]:
    data: List[T]

    fn __init__(inout self):
        self.data = List[T]()

    fn len(self) -> Int:
        return self.data.len()

    fn push(inout self, v: T):
        self.data.append(v)

    fn get(self, i: Int) -> T:
        return self.data[i]

    fn set(inout self, i: Int, v: T):
        self.data[i] = v
