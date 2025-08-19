# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


struct DataType(Copyable, Movable, EqualityComparable):
    var name: String
    fn __init__(out self, name: String):
        self.name = name
    fn __eq__(self, other: DataType) -> Bool:
        return self.name == other.name
    fn __ne__(self, other: DataType) -> Bool:
        return not self.__eq__(other)

fn INT32() -> DataType: return DataType("int32")
fn INT64() -> DataType: return DataType("int64")
fn FLOAT32() -> DataType: return DataType("float32")
fn FLOAT64() -> DataType: return DataType("float64")
fn BOOL()   -> DataType: return DataType("bool")
fn STRING() -> DataType: return DataType("string")
fn DATE64() -> DataType: return DataType("date64")
fn TIMESTAMP() -> DataType: return DataType("timestamp")
fn UNKNOWN() -> DataType: return DataType("unknown")

struct Field(Copyable, Movable, EqualityComparable):
    var name: String
    var dtype: DataType
    var nullable: Bool
    fn __init__(out self, name: String, dtype: DataType, nullable: Bool = True):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
    fn __eq__(self, other: Field) -> Bool:
        return self.name == other.name and self.dtype == other.dtype and self.nullable == other.nullable
    fn __ne__(self, other: Field) -> Bool:
        return not self.__eq__(other)

struct Schema(Copyable, Movable, EqualityComparable, Sized):
    var fields: List[Field]
    fn __init__(out self, fields: List[Field]):
        self.fields = fields
    fn __len__(self) -> Int:
        return len(self.fields)
    fn __eq__(self, other: Schema) -> Bool:
        if len(self.fields) != len(other.fields): return False
        var i = 0
        while i < len(self.fields):
            if self.fields[i] != other.fields[i]: return False
            i += 1
        return True
    fn __ne__(self, other: Schema) -> Bool:
        return not self.__eq__(other)
