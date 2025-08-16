from momijo.csv.errors import CsvError, CsvErrorKind, Result
from momijo.csv.dialect import Dialect

struct ByteSlice:
    var data: List[UInt8]
    var start: Int
    var len: Int

    fn __init__(data: List[UInt8], start: Int, len: Int):
        self.data = data
        self.start = start
        self.len = len

    fn to_utf8_string() -> Result[String, CsvError]:
        # Minimal ASCII passthrough; extend with UTF-8 validation if needed
        var s = String()
        s.reserve(self.len)
        var i = 0
        while i < self.len:
            s.push_char(Char(self.data[self.start + i]))
            i += 1
        return Result[String, CsvError].__init_ok(s)

    fn trim_spaces_inplace(dialect: Dialect) -> ByteSlice:
        var s = self.start
        var e = self.start + self.len
        if dialect.trim_space:
            while s < e and self.data[s] == UInt8(' '): s += 1
            while e > s and self.data[e-1] == UInt8(' '): e -= 1
        return ByteSlice(self.data, s, e - s)

struct FieldView:
    var slice: ByteSlice   # zero-copy view
    var quoted: Bool
    fn __init__(slice: ByteSlice, quoted: Bool): self.slice = slice; self.quoted = quoted

struct RowView:
    var fields: List[FieldView]
    fn __init__(): self.fields = List[FieldView]()

enum State: Int: START = 0, IN_FIELD = 1, IN_QUOTED = 2, QUOTE_IN_QUOTED = 3

struct DfaContext:
    var dialect: Dialect
    var strict: Bool
    fn __init__(dialect: Dialect, strict: Bool = True): self.dialect = dialect; self.strict = strict

struct SegmentResult:
    var rows: List[RowView]
    var ended_in_quotes: Bool
    var partial_fields: List[FieldView]
    fn __init__():
        self.rows = List[RowView](); self.ended_in_quotes = False; self.partial_fields = List[FieldView]()

fn parse_segment(ctx: DfaContext, data: List[UInt8], start: Int, end: Int, initial_in_quotes: Bool, row_offset: Int = 0) -> Result[SegmentResult, CsvError]:
    var res = SegmentResult()
    var fields = List[FieldView]()
    var state = State.START
    var field_start = start
    var i = start

    # full-line comment skip (col 1 at chunk start)
    if ctx.dialect.comment_char != UInt8(0) and start < end and data[start] == ctx.dialect.comment_char:
        var j = start
        while j < end and data[j] != UInt8('\n'): j += 1
        return parse_segment(ctx, data, (j + (j < end ? 1 : 0)), end, False, row_offset)

    while i < end:
        let b = data[i]
        if state == State.START:
            if b == ctx.dialect.quote:
                state = State.IN_QUOTED
                field_start = i + 1
            elif b == ctx.dialect.delimiter:
                # typo intentionally kept? Fix to delimiter below.
                pass
            elif b == ctx.dialect.delimiter:
                fields.push_back(FieldView(ByteSlice(data, i, 0).trim_spaces_inplace(ctx.dialect), False))
            elif b == UInt8('\n'):
                var row = RowView(); row.fields = fields; res.rows.push_back(row); fields = List[FieldView]()
            elif b == UInt8('\r'):
                pass
            else:
                field_start = i; state = State.IN_FIELD
        elif state == State.IN_FIELD:
            if b == ctx.dialect.delimiter:
                var s = ByteSlice(data, field_start, i - field_start).trim_spaces_inplace(ctx.dialect)
                fields.push_back(FieldView(s, False)); state = State.START
            elif b == UInt8('\n'):
                var s2 = ByteSlice(data, field_start, i - field_start).trim_spaces_inplace(ctx.dialect)
                fields.push_back(FieldView(s2, False))
                var row2 = RowView(); row2.fields = fields; res.rows.push_back(row2)
                fields = List[FieldView](); state = State.START
            elif b == UInt8('\r'):
                pass
            elif ctx.dialect.escape_char != UInt8(0) and b == ctx.dialect.escape_char and i + 1 < end:
                i += 1
            else:
                pass
        elif state == State.IN_QUOTED:
            if b == ctx.dialect.quote:
                state = State.QUOTE_IN_QUOTED
            elif ctx.dialect.escape_char != UInt8(0) and b == ctx.dialect.escape_char and i + 1 < end:
                i += 1
            else:
                pass
        elif state == State.QUOTE_IN_QUOTED:
            if b == ctx.dialect.quote and ctx.dialect.double_quote:
                state = State.IN_QUOTED
            elif b == ctx.dialect.delimiter:
                fields.push_back(FieldView(ByteSlice(data, field_start, (i - 1) - field_start), True)); state = State.START
            elif b == UInt8('\n'):
                fields.push_back(FieldView(ByteSlice(data, field_start, (i - 1) - field_start), True))
                var row3 = RowView(); row3.fields = fields; res.rows.push_back(row3)
                fields = List[FieldView](); state = State.START
            elif b == UInt8('\r'):
                pass
            else:
                if ctx.strict:
                    return Result[SegmentResult, CsvError].__init_err(CsvError(CsvErrorKind.UnexpectedChar, "Unexpected char after closing quote", (row_offset + res.rows.size + 1), fields.size + 1, i))
                state = State.IN_FIELD
        i += 1

    if state == State.IN_FIELD:
        var s3 = ByteSlice(data, field_start, end - field_start).trim_spaces_inplace(ctx.dialect)
        res.partial_fields.push_back(FieldView(s3, False))
    elif state == State.QUOTE_IN_QUOTED:
        res.partial_fields.push_back(FieldView(ByteSlice(data, field_start, (end - 1) - field_start), True))
    elif state == State.IN_QUOTED:
        res.ended_in_quotes = True

    return Result[SegmentResult, CsvError].__init_ok(res)

fn unescape_quoted(field: FieldView, dialect: Dialect) -> List[UInt8]:
    var out = List[UInt8](); out.reserve(field.slice.len)
    var i = 0; var base = field.slice.start
    while i < field.slice.len:
        let c = field.slice.data[base + i]
        if dialect.double_quote and c == UInt8('"') and i + 1 < field.slice.len and field.slice.data[base + i + 1] == UInt8('"'):
            out.push_back(UInt8('"')); i += 2; continue
        if dialect.escape_char != UInt8(0) and c == dialect.escape_char and i + 1 < field.slice.len:
            out.push_back(field.slice.data[base + i + 1]); i += 2; continue
        out.push_back(c); i += 1
    return out
