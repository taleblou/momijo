# Error types with precise location and kinds

enum CsvErrorKind: Int:
    UnexpectedChar = 1
    UnterminatedQuote = 2
    InvalidUtf8 = 3
    IndexOutOfRange = 4
    Io = 5
    ParseInt = 6
    ParseFloat = 7
    Internal = 999

struct CsvError:
    var kind: CsvErrorKind
    var message: String
    var row: Int  # 1-based; -1 unknown
    var col: Int  # 1-based; -1 unknown
    var offset: Int  # byte offset; -1 unknown

    fn __init__(kind: CsvErrorKind, message: String, row: Int = -1, col: Int = -1, offset: Int = -1):
        self.kind = kind
        self.message = message
        self.row = row
        self.col = col
        self.offset = offset

# Lightweight Result emulation (simplified)
struct Result[T, E]:
    var ok: Bool
    var value: T
    var error: E

    fn __init_ok(value: T) -> Self:
        var r = Self()
        r.ok = True
        r.value = value
        return r

    fn __init_err(error: E) -> Self:
        var r = Self()
        r.ok = False
        r.error = error
        return r
