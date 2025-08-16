from momijo.csv.dialect import Dialect
from momijo.csv.errors import CsvError, CsvErrorKind, Result
from momijo.csv.dfa_core import DfaContext, RowView, FieldView, ByteSlice, parse_segment, unescape_quoted
from momijo.csv.chunker import Chunk, make_chunks, compute_chunk_parity
from momijo.csv.simd_scan import scan_scalar

struct CsvReaderOptions:
    var dialect: Dialect
    var chunk_size: Int = 16 * 1024 * 1024
    var strict: Bool = True
    var require_final_newline: Bool = False

    fn __init__(dialect: Dialect = Dialect(), chunk_size: Int = 16 * 1024 * 1024, strict: Bool = True, require_final_newline: Bool = False):
        self.dialect = dialect
        self.chunk_size = chunk_size
        self.strict = strict
        self.require_final_newline = require_final_newline

struct ByteSource:
    var data: List[UInt8]
    fn __init__(data: List[UInt8]): self.data = data

@value
struct CsvReader:
    var src: ByteSource
    var opts: CsvReaderOptions
    var chunks: List[Chunk]
    var chunk_idx: Int
    var chunk_prefix_parity: List[Bool]
    var carry_fields: List[FieldView]
    var current_rows: List[RowView]
    var row_idx_in_chunk: Int
    var headers: List[String]
    var row_counter: Int

    fn __init__(src: ByteSource, opts: CsvReaderOptions = CsvReaderOptions()):
        self.src = src
        self.opts = opts
        self.chunks = make_chunks(src.data.size, opts.chunk_size)
        self.chunk_idx = 0
        self.chunk_prefix_parity = compute_chunk_parity(src.data, self.chunks, opts.dialect)
        self.carry_fields = List[FieldView]()
        self.current_rows = List[RowView]()
        self.row_idx_in_chunk = 0
        self.headers = List[String]()
        self.row_counter = 0

    fn refill_rows_if_needed() -> Result[Bool, CsvError]:
        if self.row_idx_in_chunk < self.current_rows.size:
            return Result[Bool, CsvError].__init_ok(True)

        if self.chunk_idx >= self.chunks.size:
            if self.carry_fields.size > 0:
                if self.require_final_newline and self.opts.strict:
                    return Result[Bool, CsvError].__init_err(CsvError(CsvErrorKind.UnterminatedQuote, "No terminal newline at EOF"))
                var last = RowView(); last.fields = self.carry_fields
                self.carry_fields = List[FieldView]()
                self.current_rows = [last]
                self.row_idx_in_chunk = 0
                return Result[Bool, CsvError].__init_ok(True)
            return Result[Bool, CsvError].__init_ok(False)

        let ch = self.chunks[self.chunk_idx]
        self.chunk_idx += 1

        let _ = scan_scalar(self.src.data, ch.start, ch.end, self.opts.dialect)
        let ctx = DfaContext(self.opts.dialect, self.opts.strict)
        let r = parse_segment(ctx, self.src.data, ch.start, ch.end, self.chunk_prefix_parity[self.chunk_idx - 1], self.row_counter)
        if not r.ok: return Result[Bool, CsvError].__init_err(r.error)
        var seg = r.value

        self.current_rows = List[RowView]()
        if seg.rows.size > 0 and self.carry_fields.size > 0:
            var first = seg.rows[0]
            var merged = RowView(); merged.fields = List[FieldView]()
            for f in self.carry_fields: merged.fields.push_back(f)
            for f2 in first.fields: merged.fields.push_back(f2)
            self.current_rows.push_back(merged)
            var idx = 1
            while idx < seg.rows.size:
                self.current_rows.push_back(seg.rows[idx]); idx += 1
            self.carry_fields = List[FieldView]()
        else:
            for rr in seg.rows: self.current_rows.push_back(rr)

        if seg.ended_in_quotes:
            if self.current_rows.size > 0:
                var last_row = self.current_rows.back()
                self.current_rows.pop_back()
                for f in last_row.fields: self.carry_fields.push_back(f)
        else:
            if seg.partial_fields.size > 0:
                for pf in seg.partial_fields: self.carry_fields.push_back(pf)

        self.row_idx_in_chunk = 0
        self.row_counter += self.current_rows.size
        return Result[Bool, CsvError].__init_ok(self.current_rows.size > 0)

    fn read_headers_if_needed() -> Result[Void, CsvError]:
        if not self.opts.dialect.has_header or self.headers.size > 0:
            return Result[Void, CsvError].__init_ok(Void())
        let row_res = self.next_row()
        if not row_res.ok: return Result[Void, CsvError].__init_err(row_res.error)
        let row = row_res.value
        var h = List[String]()
        for f in row.fields:
            let s = f.slice.to_utf8_string()
            if not s.ok: return Result[Void, CsvError].__init_err(s.error)
            h.push_back(s.value)
        self.headers = h
        return Result[Void, CsvError].__init_ok(Void())

    fn next_row() -> Result[RowView, CsvError]:
        let has = self.refill_rows_if_needed()
        if not has.ok: return Result[RowView, CsvError].__init_err(has.error)
        if not has.value: return Result[RowView, CsvError].__init_err(CsvError(CsvErrorKind.Io, "EOF"))
        let row = self.current_rows[self.row_idx_in_chunk]
        self.row_idx_in_chunk += 1
        return Result[RowView, CsvError].__init_ok(row)
