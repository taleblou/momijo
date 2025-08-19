# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


struct Offsets(Copyable, Movable, EqualityComparable, Sized):
    var data: List[Int]

    fn __init__(out self):
        self.data = List[Int]()
        self.data.append(0)

    fn __copyinit__(out self, other: Offsets):
        self.data = List[Int]()
        var i = 0
        while i < len(other.data):
            self.data.append(other.data[i])
            i += 1

    fn __len__(self) -> Int:
        return len(self.data)

    fn __eq__(self, other: Offsets) -> Bool:
        if len(self.data) != len(other.data): return False
        var i = 0
        while i < len(self.data):
            if self.data[i] != other.data[i]: return False
            i += 1
        return True

    fn __ne__(self, other: Offsets) -> Bool:
        return not self.__eq__(other)

    fn append_offset(mut self, o: Int):
        self.data.append(o)

    fn len(self) -> Int:
        return len(self.data)

    fn last(self) -> Int:
        return self.data[len(self.data) - 1]

    fn add_length(mut self, length: Int):
        var next = self.last() + length
        self.data.append(next)

    fn is_valid(self) -> Bool:
        var i = 1
        while i < len(self.data):
            if self.data[i] < self.data[i - 1]:
                return False
            i += 1
        return True

fn from_lengths(lengths: List[Int]) -> Offsets:
    var o = Offsets()
    var i = 0
    while i < len(lengths):
        o.add_length(lengths[i])
        i += 1
    return o
