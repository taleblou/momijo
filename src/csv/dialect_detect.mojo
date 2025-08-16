from momijo.csv.dialect import Dialect

fn detect_delimiter(bytes: List[UInt8], candidates: List[UInt8]) -> UInt8:
    var best_delim = UInt8(','); var best_score = -1
    for d in candidates:
        var commas = 0; var lines = 0; var i = 0
        while i < min(bytes.size, 1_000_000):
            let b = bytes[i]
            if b == d: commas += 1
            if b == UInt8('\n'): lines += 1
            i += 1
        var score = (lines > 0 ? (commas / max(1, lines)) : 0)
        if score > best_score: best_score = score; best_delim = d
    return best_delim

fn auto_dialect(bytes: List[UInt8]) -> Dialect:
    var cand = [UInt8(','), UInt8(';'), UInt8('\t'), UInt8('|')]
    var d = detect_delimiter(bytes, cand)
    return Dialect(d, UInt8('"'), UInt8('\n'), True, UInt8(0), False, False, UInt8(0), False)
