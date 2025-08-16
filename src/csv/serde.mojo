from momijo.csv.reader import RowView
from momijo.csv.dfa_core import FieldView, unescape_quoted
from momijo.csv.errors import CsvError, CsvErrorKind, Result
from momijo.csv.dialect import Dialect

trait FromCsvRow[T]:
    static fn from_row(row: RowView, dialect: Dialect) -> Result[T, CsvError]

fn field_to_string(dialect: Dialect, f: FieldView) -> Result[String, CsvError]:
    if f.quoted:
        let buf = unescape_quoted(f, dialect)
        var s = String(); s.reserve(buf.size)
        for b in buf: s.push_char(Char(b))
        return Result[String, CsvError].__init_ok(s)
    else:
        return f.slice.to_utf8_string()

fn parse_int32(dialect: Dialect, f: FieldView) -> Result[Int32, CsvError]:
    let sres = field_to_string(dialect, f)
    if not sres.ok: return Result[Int32, CsvError].__init_err(sres.error)
    let s = sres.value
    var i = 0; var sign: Int32 = 1
    if s.size > 0 and s[0] == '-': sign = -1; i = 1
    var acc: Int32 = 0
    while i < s.size:
        let ch = s[i]
        if ch < '0' or ch > '9': return Result[Int32, CsvError].__init_err(CsvError(CsvErrorKind.ParseInt, "Invalid int"))
        acc = acc * 10 + Int32(ch - '0'); i += 1
    return Result[Int32, CsvError].__init_ok(sign * acc)

fn parse_float64(dialect: Dialect, f: FieldView) -> Result[Float64, CsvError]:
    let sres = field_to_string(dialect, f)
    if not sres.ok: return Result[Float64, CsvError].__init_err(sres.error)
    let s = sres.value
    var val: Float64 = 0.0; var dot = -1; var sign: Float64 = 1.0; var i = 0
    if s.size > 0 and s[0] == '-': sign = -1.0; i = 1
    while i < s.size:
        let ch = s[i]
        if ch == '.': dot = i
        elif ch >= '0' and ch <= '9': val = val * 10.0 + Float64(ch - '0')
        i += 1
    if dot >= 0:
        let frac_len = s.size - (dot + 1)
        var scale: Float64 = 1.0; var k = 0
        while k < frac_len: scale *= 10.0; k += 1
        val = val / scale
    return Result[Float64, CsvError].__init_ok(sign * val)

# Example struct
struct Person:
    var name: String
    var age: Int32
    var height: Float64
    fn __init__(name: String, age: Int32, height: Float64): self.name = name; self.age = age; self.height = height

impl FromCsvRow[Person]:
    static fn from_row(row: RowView, dialect: Dialect) -> Result[Person, CsvError]:
        if row.fields.size < 3:
            return Result[Person, CsvError].__init_err(CsvError(CsvErrorKind.IndexOutOfRange, "Need 3 fields"))
        let n = field_to_string(dialect, row.fields[0]); if not n.ok: return Result[Person, CsvError].__init_err(n.error)
        let a = parse_int32(dialect, row.fields[1]);    if not a.ok: return Result[Person, CsvError].__init_err(a.error)
        let h = parse_float64(dialect, row.fields[2]);  if not h.ok: return Result[Person, CsvError].__init_err(h.error)
        return Result[Person, CsvError].__init_ok(Person(n.value, a.value, h.value))

# Header helper (linear search)
fn find_index(headers: List[String], name: String) -> Int:
    var i = 0
    while i < headers.size:
        if headers[i] == name: return i
        i += 1
    return -1
