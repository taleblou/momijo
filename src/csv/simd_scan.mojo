# SIMD-friendly prescan producing positions of key bytes and a quote-parity mask.
# Scalar path works now; vectorization can replace inner loops.

from momijo.csv.dialect import Dialect

struct ScanMasks:
    var delim_positions: List[Int]
    var nl_positions: List[Int]
    var quote_positions: List[Int]

    fn __init__():
        self.delim_positions = List[Int]()
        self.nl_positions = List[Int]()
        self.quote_positions = List[Int]()

fn is_crlf_pair(bytes: List[UInt8], i: Int) -> Bool:
    if i + 1 < bytes.size and bytes[i] == UInt8('\r') and bytes[i+1] == UInt8('\n'):
        return True
    return False

fn scan_scalar(bytes: List[UInt8], start: Int, end: Int, dialect: Dialect) -> ScanMasks:
    var masks = ScanMasks()
    var i = start
    while i < end:
        let b = bytes[i]
        if b == dialect.delimiter:
            masks.delim_positions.push_back(i)
        elif b == UInt8('\n') or b == UInt8('\r'):
            if b == UInt8('\r') and is_crlf_pair(bytes, i):
                masks.nl_positions.push_back(i + 1)  # mark LF index
                i += 1
            else:
                masks.nl_positions.push_back(i)
        elif b == dialect.quote:
            masks.quote_positions.push_back(i)
        i += 1
    return masks

# Build in-quotes boolean mask (approx parity). Double-quotes handled by DFA later.
fn build_in_quotes_mask(length: Int, quote_positions: List[Int], base_parity: Bool) -> List[Bool]:
    var inq = List[Bool](length)
    var parity = base_parity
    var qi = 0
    var next_q = -1
    if quote_positions.size > 0:
        next_q = quote_positions[0]
    for i in range(0, length):
        if next_q == i:
            parity = not parity
            qi += 1
            if qi < quote_positions.size: next_q = quote_positions[qi]
            else: next_q = -1
        inq[i] = parity
    return inq
