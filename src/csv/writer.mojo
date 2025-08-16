from momijo.csv.dialect import Dialect
from momijo.csv.errors import CsvError, CsvErrorKind, Result

enum QuotePolicy: Int: MINIMAL = 0, ALL = 1, NON_NUMERIC = 2

struct CsvWriterOptions:
    var dialect: Dialect
    var quote_policy: QuotePolicy = QuotePolicy.MINIMAL
    fn __init__(dialect: Dialect = Dialect(), quote_policy: QuotePolicy = QuotePolicy.MINIMAL):
        self.dialect = dialect; self.quote_policy = quote_policy

struct ByteSink:
    var data: List[UInt8]
    fn __init__(): self.data = List[UInt8]()
    fn as_string() -> String:
        var s = String(); s.reserve(self.data.size)
        for b in self.data: s.push_char(Char(b))
        return s

@value
struct CsvWriter:
    var sink: ByteSink
    var opts: CsvWriterOptions
    fn __init__(sink: ByteSink, opts: CsvWriterOptions = CsvWriterOptions()): self.sink = sink; self.opts = opts

    fn needs_quote(field: List[UInt8]) -> Bool:
        if self.opts.quote_policy == QuotePolicy.ALL: return True
        if self.opts.quote_policy == QuotePolicy.NON_NUMERIC:
            if field.size == 0: return True
            var i = 0; if field[0] == UInt8('-'): i = 1
            var numeric = True
            while i < field.size:
                let b = field[i]
                if b == UInt8('.') or (b >= UInt8('0') and b <= UInt8('9')): pass
                else: numeric = False; break
                i += 1
            return not numeric
        for b in field:
            if b == self.opts.dialect.delimiter or b == self.opts.dialect.quote or b == UInt8('\n') or b == UInt8('\r') or b == UInt8(' '):
                return True
        return False

    fn write_escaped(field: List[UInt8]):
        for b in field:
            if b == self.opts.dialect.quote and self.opts.dialect.double_quote:
                self.sink.data.push_back(self.opts.dialect.quote); self.sink.data.push_back(self.opts.dialect.quote)
            elif self.opts.dialect.escape_char != UInt8(0) and b == self.opts.dialect.escape_char:
                self.sink.data.push_back(self.opts.dialect.escape_char); self.sink.data.push_back(b)
            else:
                self.sink.data.push_back(b)

    fn write_row_bytes(fields: List[List[UInt8]]) -> Result[Void, CsvError]:
        var first = True
        for f in fields:
            if not first: self.sink.data.push_back(self.opts.dialect.delimiter)
            first = False
            if self.needs_quote(f):
                self.sink.data.push_back(self.opts.dialect.quote); self.write_escaped(f); self.sink.data.push_back(self.opts.dialect.quote)
            else:
                for b in f: self.sink.data.push_back(b)
        if self.opts.dialect.lineterminator == UInt8('\n'):
            self.sink.data.push_back(UInt8('\n'))
        else:
            self.sink.data.push_back(UInt8('\r')); self.sink.data.push_back(UInt8('\n'))
        return Result[Void, CsvError].__init_ok(Void())

fn write_row_strings(self: inout CsvWriter, fields: List[String]) -> Result[Void, CsvError]:
    var bufs = List[List[UInt8]]()
    var i = 0
    while i < fields.size:
        var bs = List[UInt8]()
        for ch in fields[i]: bs.push_back(UInt8(ch))
        bufs.push_back(bs); i += 1
    return self.write_row_bytes(bufs)
