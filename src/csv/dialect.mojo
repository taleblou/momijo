# Dialect configuration + policies (inspired by Python csv, RFC 4180, and common variants)

struct Dialect:
    var delimiter: UInt8 = UInt8(',')
    var quote: UInt8 = UInt8('"')
    var lineterminator: UInt8 = UInt8('\n')  # emit LF; CRLF recognized on read
    var double_quote: Bool = True             # "" inside quoted fields
    var escape_char: UInt8 = UInt8(0)         # 0=disabled; if nonzero, escape this char
    var allow_loose_quotes: Bool = False      # relaxed parsing
    var trim_space: Bool = False              # trim leading/trailing spaces outside quotes
    var comment_char: UInt8 = UInt8(0)        # 0=disabled; if nonzero, lines starting with it are comments
    var has_header: Bool = False              # if true, first row is header

    fn __init__(delimiter: UInt8 = UInt8(','), quote: UInt8 = UInt8('"'),
                lineterminator: UInt8 = UInt8('\n'), double_quote: Bool = True,
                escape_char: UInt8 = UInt8(0), allow_loose_quotes: Bool = False,
                trim_space: Bool = False, comment_char: UInt8 = UInt8(0), has_header: Bool = False):
        self.delimiter = delimiter
        self.quote = quote
        self.lineterminator = lineterminator
        self.double_quote = double_quote
        self.escape_char = escape_char
        self.allow_loose_quotes = allow_loose_quotes
        self.trim_space = trim_space
        self.comment_char = comment_char
        self.has_header = has_header

# Preset dialects
fn rfc4180() -> Dialect: return Dialect(UInt8(','), UInt8('"'), UInt8('\n'), True, UInt8(0), False, False, UInt8(0), False)
fn excel_csv() -> Dialect: return Dialect(UInt8(','), UInt8('"'), UInt8('\r'), True, UInt8(0), False, False, UInt8(0), True)
fn tsv() -> Dialect: return Dialect(UInt8('\t'), UInt8('"'), UInt8('\n'), True, UInt8(0), False, False, UInt8(0), False)
fn pipe_sep() -> Dialect: return Dialect(UInt8('|'), UInt8('"'), UInt8('\n'), True, UInt8(0), False, False, UInt8(0), False)
