from collections.list import List
from pathlib import Path

# ----- String -> Bytes (ASCII-safe) -----
fn str_to_bytes(s: String) -> List[UInt8]:
    # ASCII-safe: emit bytes 0..127; otherwise '?'
    var out = List[UInt8]()
    for cp in s.codepoints():
        var code = Int(cp)
        if code >= 0 and code < 128:
            out.append(UInt8(code))
        else:
            out.append(UInt8(63))  # '?'
    return out.copy() 

@always_inline
fn string_to_bytes(s: String) -> List[UInt8]:
    return str_to_bytes(s)

@always_inline
fn ascii_byte_to_string(v: Int) -> String:
    var tbl = List[String]()
    # Printable ASCII 32..126
    tbl.append(" "); tbl.append("!"); tbl.append("\""); tbl.append("#"); tbl.append("$"); tbl.append("%"); tbl.append("&"); tbl.append("'"); tbl.append("("); tbl.append(")"); tbl.append("*"); tbl.append("+"); tbl.append(","); tbl.append("-"); tbl.append("."); tbl.append("/")
    tbl.append("0"); tbl.append("1"); tbl.append("2"); tbl.append("3"); tbl.append("4"); tbl.append("5"); tbl.append("6"); tbl.append("7"); tbl.append("8"); tbl.append("9")
    tbl.append(":"); tbl.append(";"); tbl.append("<"); tbl.append("="); tbl.append(">"); tbl.append("?"); tbl.append("@")
    tbl.append("A"); tbl.append("B"); tbl.append("C"); tbl.append("D"); tbl.append("E"); tbl.append("F"); tbl.append("G"); tbl.append("H"); tbl.append("I"); tbl.append("J"); tbl.append("K"); tbl.append("L"); tbl.append("M"); tbl.append("N"); tbl.append("O"); tbl.append("P"); tbl.append("Q"); tbl.append("R"); tbl.append("S"); tbl.append("T"); tbl.append("U"); tbl.append("V"); tbl.append("W"); tbl.append("X"); tbl.append("Y"); tbl.append("Z")
    tbl.append("["); tbl.append("\\"); tbl.append("]"); tbl.append("^"); tbl.append("_"); tbl.append("`")
    tbl.append("a"); tbl.append("b"); tbl.append("c"); tbl.append("d"); tbl.append("e"); tbl.append("f"); tbl.append("g"); tbl.append("h"); tbl.append("i"); tbl.append("j"); tbl.append("k"); tbl.append("l"); tbl.append("m"); tbl.append("n"); tbl.append("o"); tbl.append("p"); tbl.append("q"); tbl.append("r"); tbl.append("s"); tbl.append("t"); tbl.append("u"); tbl.append("v"); tbl.append("w"); tbl.append("x"); tbl.append("y"); tbl.append("z")
    tbl.append("{"); tbl.append("|"); tbl.append("}"); tbl.append("~")

    if v >= 32 and v <= 126:
        return tbl[v - 32]
    elif v == 10:
        return String("\n")
    elif v == 13:
        return String("\r")
    elif v == 9:
        return String("\t")
    else:
        return String("?")

        
fn bytes_to_string(b: List[UInt8]) -> String:
    # ASCII-safe: map 0..127 to chars; others become '?'
    var out = String("")
    for by in b:
        var v = Int(by)
        if v >= 0 and v < 128:
            out += ascii_byte_to_string(v)
        else:
            out += String("?")
    return out

# ----- File IO -----
fn write_bytes(path: String, data: List[UInt8]) -> Bool:
    try:
        var f = open(Path(path), "w")
        var s = bytes_to_string(data)
        _ = f.write(s)
        f.close()
        return True
    except _:
        return False

fn read_bytes(path: String) -> List[UInt8]:
    try:
        var f = open(Path(path), "r")
        var s = f.read()
        f.close()
        return string_to_bytes(s)
    except _:
        return List[UInt8]()  

@always_inline
fn u8_to_char(b: UInt8) -> String:
    return ascii_byte_to_string(Int(b))