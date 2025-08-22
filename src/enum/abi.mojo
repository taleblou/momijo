# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# ---------- ASCII helpers ----------

fn _ascii_tolower(c: UInt8) -> UInt8:
    # 'A'..'Z' => 'a'..'z'
    if c >= UInt8(65) and c <= UInt8(90):
        return c + UInt8(32)
    return c

fn ascii_lower(s: String) -> String:
    let bs = s.bytes()
    var out = List[UInt8]()
    for b in bs:
        out.append(_ascii_tolower(b))
    return String(out)

# ---------- Name ↔ Index mapping ----------

# Returns the name at index i, or empty string if out of range.
fn enum_name_at(names: List[String], i: Int) -> String:
    if i < 0 or i >= len(names):
        return ""
    return names[i]

# Returns index of token in names, or -1 if not found.
fn enum_index_of(names: List[String], token: String, case_sensitive: Bool = True) -> Int:
    if case_sensitive:
        var i = 0
        while i < len(names):
            if names[i] == token:
                return i
            i += 1
        return -1
    else:
        let t = ascii_lower(token)
        var i = 0
        while i < len(names):
            if ascii_lower(names[i]) == t:
                return i
            i += 1
        return -1

# Validate an index for a list of names.
fn enum_index_valid(names: List[String], i: Int) -> Bool:
    return i >= 0 and i < len(names)

# Clamp index into range [0, len-1]. If names empty, returns -1.
fn enum_index_clamp(names: List[String], i: Int) -> Int:
    let n = len(names)
    if n == 0:
        return -1
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i

# ---------- Integer (de)serialization ----------

fn to_i8(i: Int) -> Int8:  return Int8(i)
fn to_i16(i: Int) -> Int16: return Int16(i)
fn to_i32(i: Int) -> Int32: return Int32(i)
fn to_i64(i: Int) -> Int64: return Int64(i)

fn from_i8(x: Int8) -> Int:  return Int(x)
fn from_i16(x: Int16) -> Int: return Int(x)
fn from_i32(x: Int32) -> Int: return Int(x)
fn from_i64(x: Int64) -> Int: return Int(x)

# Safe load ensuring result is a valid index for given names.
fn enum_index_from_i32(names: List[String], x: Int32, default_on_error: Int = -1) -> Int:
    let i = from_i32(x)
    return i if enum_index_valid(names, i) else default_on_error

fn enum_index_from_i64(names: List[String], x: Int64, default_on_error: Int = -1) -> Int:
    let i = from_i64(x)
    return i if enum_index_valid(names, i) else default_on_error

# ---------- Bit-mask enum sets (up to 64 members) ----------

# Build a bit mask with bit i set. If i≥64, returns 0.
fn enum_bit(i: Int) -> UInt64:
    if i < 0 or i >= 64:
        return UInt64(0)
    return UInt64(1) << UInt64(i)

fn enumset_empty() -> UInt64:
    return UInt64(0)

fn enumset_full(count: Int) -> UInt64:
    if count <= 0:
        return UInt64(0)
    if count >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(count)) - UInt64(1)

fn enumset_has(mask: UInt64, i: Int) -> Bool:
    if i < 0 or i >= 64:
        return False
    return (mask & enum_bit(i)) != UInt64(0)

fn enumset_add(mask: UInt64, i: Int) -> UInt64:
    return mask | enum_bit(i)

fn enumset_remove(mask: UInt64, i: Int) -> UInt64:
    return mask & (~enum_bit(i))

fn enumset_toggle(mask: UInt64, i: Int) -> UInt64:
    return mask ^ enum_bit(i)

fn enumset_count(mask: UInt64) -> Int:
    var x = mask
    var c: Int = 0
    while x != UInt64(0):
        c += 1
        x = x & (x - UInt64(1))
    return c

# Map an index list into a mask (indices outside [0,63] are ignored).
fn enumset_from_indices(idxs: List[Int]) -> UInt64:
    var m = UInt64(0)
    for i in idxs:
        m = enumset_add(m, i)
    return m

# Expand a mask back to a sorted list of indices.
fn enumset_to_indices(mask: UInt64) -> List[Int]:
    var out = List[Int]()
    var i: Int = 0
    while i < 64:
        if enumset_has(mask, i):
            out.append(i)
        i += 1
    return out

# Intersect mask with valid range [0,count)
fn enumset_sanitize(mask: UInt64, count: Int) -> UInt64:
    return mask & enumset_full(count)