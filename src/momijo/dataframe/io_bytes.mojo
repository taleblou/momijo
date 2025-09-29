# Project:      Momijo
# Module:       dataframe.io_bytes
# File:         io_bytes.mojo
# Path:         dataframe/io_bytes.mojo
#
# Description:  dataframe.io_bytes — Io Bytes module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: —
#   - Key functions: str_to_bytes, bytes_to_string, ascii_byte_to_string, write_bytes, read_bytes

from collections.list import List
from pathlib.path import Path 


fn str_to_bytes(s: String) -> List[UInt8]:
# ASCII-safe: emit bytes 0..127; otherwise use '?' (63)
    var out = List[UInt8]()
    for ch in s:
        var code: Int = 0
        try:
            code = Int(ch)
        except:
            code = 0          # implementation-defined; treat as code point
        if code >= 0 and code < 128:
            out.append(UInt8(code))
        else:
            out.append(UInt8(63))# '?' 
    return out

fn bytes_to_string(b: List[UInt8]) -> String:
# ASCII-safe: map 0..127 to chars; others become '?'
    var out = String("")
    for by in b:
        var v = Int(by)
        if v >= 0 and v < 128:
            out += ascii_byte_to_string(v)
        else:
            out += String('?')
    return out


# Safe ASCII byte -> String (single-char). Falls back to "?" if out of range.
fn ascii_byte_to_string(v: Int) -> String:
    if v < 0 or v > 127:
        return String("?")
    var bytes = List[UInt8]()
    bytes.push_back(UInt8(v))
# Prefer String.from_utf8 if available in your Mojo version; otherwise this is a minimal fallback.
    return String.from_utf8(bytes)



fn write_bytes(path: String, data: List[UInt8]) -> Bool:
    try:
        var f = open(Path(path), "w")
        var s = bytes_to_string(data)
        f.write(s)
        f.close()
        return True
    except:
        return False

fn read_bytes(path: String) -> List[UInt8]:
    try:
        var f = open(Path(path), "r")
        var s = f.read()
        f.close()
        return string_to_bytes(s)
    except:
        return List[UInt8]()