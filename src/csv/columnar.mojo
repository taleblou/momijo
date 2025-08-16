# Columnar builder for EDA/ML-like consumption with post-hoc type inference
from momijo.csv.dfa_core import RowView, FieldView
from momijo.csv.errors import CsvError, Result, CsvErrorKind
from momijo.csv.dfa_core import unescape_quoted
from momijo.csv.dialect import Dialect

enum ColumnKind: Int: String = 0, Int64 = 1, Float64 = 2

struct ColumnString: var data: List[String]; fn __init__(): self.data = List[String]()
struct ColumnI64:    var data: List[Int64];  fn __init__(): self.data = List[Int64]()
struct ColumnF64:    var data: List[Float64];fn __init__(): self.data = List[Float64]()

struct ColumnarTable:
    var headers: List[String]
    var kinds: List[ColumnKind]
    var cols_s: List[ColumnString]
    var cols_i: List[ColumnI64]
    var cols_f: List[ColumnF64]
    fn __init__():
        self.headers = List[String](); self.kinds = List[ColumnKind]()
        self.cols_s = List[ColumnString](); self.cols_i = List[ColumnI64](); self.cols_f = List[ColumnF64]()

struct ColumnarBuilder:
    var dialect: Dialect
    var headers: List[String]
    var rows_seen: Int
    fn __init__(dialect: Dialect):
        self.dialect = dialect; self.headers = List[String](); self.rows_seen = 0

    fn set_headers(h: List[String]): self.headers = h

    fn add_row(row: RowView, table: inout ColumnarTable) -> Result[Void, CsvError]:
        if table.headers.size == 0 and self.headers.size > 0: table.headers = self.headers
        while table.cols_s.size < row.fields.size: table.cols_s.push_back(ColumnString())
        var c = 0
        while c < row.fields.size:
            let f = row.fields[c]
            var s: String
            if f.quoted:
                let buf = unescape_quoted(f, self.dialect)
                var tmp = String(); tmp.reserve(buf.size)
                for b in buf: tmp.push_char(Char(b))
                s = tmp
            else:
                let sres = f.slice.to_utf8_string()
                if not sres.ok: return Result[Void, CsvError].__init_err(sres.error)
                s = sres.value
            table.cols_s[c].data.push_back(s)
            c += 1
        self.rows_seen += 1
        return Result[Void, CsvError].__init_ok(Void())

    fn finalize(table: inout ColumnarTable):
        var ncols = table.cols_s.size
        table.kinds = List[ColumnKind](ncols)
        while table.cols_i.size < ncols: table.cols_i.push_back(ColumnI64())
        while table.cols_f.size < ncols: table.cols_f.push_back(ColumnF64())

        var col = 0
        while col < ncols:
            var all_int = True; var all_num = True
            var r = 0
            while r < table.cols_s[col].data.size:
                let s = table.cols_s[col].data[r]
                if s.size == 0: r += 1; continue
                var i = 0
                if s[0] == '-': i = 1
                var is_int = True
                while i < s.size:
                    let ch = s[i]
                    if ch < '0' or ch > '9': is_int = False; break
                    i += 1
                if not is_int:
                    all_int = False
                    var j = 0; var seen_dot = False; var seen_digit = False
                    if s[0] == '-': j = 1
                    while j < s.size:
                        let ch2 = s[j]
                        if ch2 == '.': 
                            if seen_dot: all_num = False; break
                            seen_dot = True
                        elif ch2 >= '0' and ch2 <= '9': seen_digit = True
                        else: all_num = False; break
                        j += 1
                    if not seen_digit: all_num = False
                r += 1

            if all_int:
                table.kinds[col] = ColumnKind.Int64
                var rr = 0
                while rr < table.cols_s[col].data.size:
                    let s2 = table.cols_s[col].data[rr]
                    if s2.size == 0: table.cols_i[col].data.push_back(Int64(0))
                    else:
                        var k = 0; var sign: Int64 = 1
                        if s2[0] == '-': sign = -1; k = 1
                        var acc: Int64 = 0
                        while k < s2.size: acc = acc * 10 + Int64(s2[k] - '0'); k += 1
                        table.cols_i[col].data.push_back(sign * acc)
                    rr += 1
            elif all_num:
                table.kinds[col] = ColumnKind.Float64
                var rr2 = 0
                while rr2 < table.cols_s[col].data.size:
                    let s3 = table.cols_s[col].data[rr2]
                    if s3.size == 0: table.cols_f[col].data.push_back(0.0)
                    else:
                        var sign2: Float64 = 1.0; var v: Float64 = 0.0; var dot = -1; var p = 0
                        if s3[0] == '-': sign2 = -1.0; p = 1
                        while p < s3.size:
                            let ch = s3[p]
                            if ch == '.': dot = p
                            elif ch >= '0' and ch <= '9': v = v * 10.0 + Float64(ch - '0')
                            p += 1
                        if dot >= 0:
                            let frac_len = s3.size - (dot + 1)
                            var scale: Float64 = 1.0; var kk = 0
                            while kk < frac_len: scale *= 10.0; kk += 1
                            v = v / scale
                        table.cols_f[col].data.push_back(sign2 * v)
                    rr2 += 1
            else:
                table.kinds[col] = ColumnKind.String
            col += 1
