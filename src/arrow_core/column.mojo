# MIT License
# Project: momijo.arrow_core
# File: momijo/arrow_core/column.mojo

from momijo.arrow_core.array import Array
from momijo.arrow_core.byte_string_array import ByteStringArray

# Generic value column backed by Array[T].
struct Column[T: Copyable & Movable](Copyable, Movable, Sized):
    var name: String
    var values: Array[T]

    # ---------- Constructors ----------
    fn __init__(out self, name: String = "", values: Array[T] = Array[T]()):
        self.name = name
        self.values = values

    # Assign content into an existing Column from a plain List[T] (all valid).
    fn assign_from_list(mut self, name: String, vals: List[T]):
        self.name = name
        var arr = Array[T]()        # build empty array
        # fill values via push to avoid alt-init quirks
        var i: Int = 0
        while i < len(vals):
            arr.push(vals[i], True)
            i += 1
        self.values = arr

    # ---------- Properties ----------
    @always_inline
    fn __len__(self) -> Int:
        return self.values.len()

    fn len(self) -> Int:
        return self.values.len()

    fn is_valid(self, i: Int) -> Bool:
        return self.values.is_valid(i)

    fn null_count(self) -> Int:
        return self.values.null_count()

    # ---------- Access ----------
    fn get(self, i: Int) -> T:
        return self.values.get(i)

    fn get_or(self, i: Int, default: T) -> T:
        return self.values.get_or(i, default)

    # ---------- Mutation ----------
    fn set(mut self, i: Int, v: T, valid: Bool = True) -> Bool:
        return self.values.set(i, v, valid)

    fn push(mut self, v: T, valid: Bool = True):
        self.values.push(v, valid)

    fn clear(mut self):
        self.values.clear()

    # ---------- Conversion ----------
    fn to_list(self) -> List[T]:
        return self.values.to_list()

    fn compact_values(self) -> List[T]:
        return self.values.compact_values()


# Specialized string column backed by ByteStringArray.
struct StringColumn(Copyable, Movable, Sized):
    var name: String
    var values: ByteStringArray

    fn __init__(out self, name: String = "", values: ByteStringArray = ByteStringArray()):
        self.name = name
        self.values = values

    @always_inline
    fn __len__(self) -> Int:
        return self.values.len()

    fn len(self) -> Int:
        return self.values.len()

    fn is_valid(self, i: Int) -> Bool:
        return self.values.is_valid(i)

    fn null_count(self) -> Int:
        var n = self.values.len()
        var s: Int = 0
        var i: Int = 0
        while i < n:
            if not self.values.is_valid(i):
                s += 1
            i += 1
        return s

    # ---------- Access ----------
    fn get(self, i: Int) -> String:
        return self.values.get(i)

    fn get_or(self, i: Int, default: String) -> String:
        return self.values.get_or(i, default)

    # ---------- Mutation ----------
    fn push(mut self, s: String, valid: Bool = True):
        self.values.push(s, valid)

    fn push_null(mut self):
        self.values.push_null()

    fn clear(mut self):
        self.values.clear()

    # ---------- Conversion ----------
    fn to_strings(self) -> List[String]:
        return self.values.to_strings()

    fn to_optional_strings(self) -> List[Optional[String]]:
        return self.values.to_optional_strings()
