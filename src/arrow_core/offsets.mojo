struct Offsets:
    var data: List[Int]

    fn __init__(out self):
        self.data = List[Int]()
        self.data.append(0)

    fn append_offset(mut self, o: Int):
        self.data.append(o)

    fn add_length(mut self, length: Int):
        self.data.append(self.last() + length)

    fn len(self) -> Int:
        return len(self.data)

    fn last(self) -> Int:
        return self.data[len(self.data) - 1]

    fn at(self, i: Int) -> Int:
        return self.data[i]

    fn is_valid(self) -> Bool:
        if len(self.data) == 0:
            return False
        if self.data[0] != 0:
            return False
        var i = 1
        while i < len(self.data):
            if self.data[i] < self.data[i - 1]:
                return False
            i += 1
        return True
