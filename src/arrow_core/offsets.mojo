# Cumulative offsets for variable-length data.
# Inputs: created by constructor.
# Returns: not applicable.
struct Offsets:
    var data: List[Int]
    fn __init__(out self):
        self.data = List[Int]()
        self.data.append(0)

# Appends a raw offset value.
# Inputs: one value to append.
# Returns: not applicable.
    fn append_offset(mut self, o: Int):
        self.data.append(o)

# Appends the previous value plus a length.
# Inputs: one length to add.
# Returns: not applicable.
    fn add_length(mut self, length: Int):
        self.data.append(self.last() + length)

# Reports the number of logical elements.
# Inputs: none.
# Returns: the count of elements.
    fn len(self) -> Int:
        return len(self.data)

# Reports the last stored offset.
# Inputs: none.
# Returns: the last value.
    fn last(self) -> Int:
        return self.data[len(self.data) - 1]

# Reads an offset at a position.
# Inputs: a position inside the list.
# Returns: the stored value.
    fn at(self, i: Int) -> Int:
        return self.data[i]

# Checks that offsets start at zero and never decrease.
# Inputs: none.
# Returns: true if valid; false otherwise.
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