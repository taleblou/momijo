# Arrow-like typing for Mojo (simplified)

struct DataType:
    name: String

    fn __init__(inout self, name: String):
        self.name = name

    fn __eq__(self, other: DataType) -> Bool:
        return self.name == other.name

let INT32 = DataType("int32")
let INT64 = DataType("int64")
let FLOAT32 = DataType("float32")
let FLOAT64 = DataType("float64")
let BOOL = DataType("bool")
let STRING = DataType("string")
let DATE64 = DataType("date64")
let TIMESTAMP = DataType("timestamp")
let UNKNOWN = DataType("unknown")

struct Field:
    name: String
    dtype: DataType
    nullable: Bool = True

struct Schema:
    fields: List[Field]

    fn __init__(inout self, fields: List[Field]):
        self.fields = fields

    fn field_index(self, name: String) -> Int:
        var i = 0
        for f in self.fields:
            if f.name == name: return i
            i += 1
        return -1

    fn field_names(self) -> List[String]:
        let names = List[String]()
        for f in self.fields:
            names.append(f.name)
        return names
