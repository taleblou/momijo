# Project:      Momijo
# Module:       src.momijo.utils.env
# File:         env.mojo
# Path:         src/momijo/utils/env.mojo
#
# Description:  General-purpose utilities and math helpers used across Momijo,
#               designed to be small, composable, and well-tested.
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
#   - Structs: Env
#   - Key functions: _lower_ascii, _is_truthy, _is_falsy, _parse_int64_safe, _parse_float64_safe, __init__, size, keys ...
#   - Uses generic functions/types with explicit trait bounds.


from stdlib.dict import Dict
from stdlib.list import List
from stdlib.string import String

fn _lower_ascii(s: String) -> String:
    # minimal ASCII-only lowercase (Mojo String is UTF-8; here we assume ASCII keys/values)
    var out = String("")
    var n = len(s)
    var i = 0
    while i < n:
        var b = s.bytes()[i]
        if b >= UInt8(65) and b <= UInt8(90):           # 'A'..'Z'
            b = UInt8(b + UInt8(32))                    # to 'a'..'z'
        out = out + String.from_utf8([b])
        i += 1
    return out

# -------------------------------
# Boolean parsing
# -------------------------------
fn _is_truthy(s: String) -> Bool:
    var v = _lower_ascii(s).strip()
    # accepted true tokens
    if v == String("1") or v == String("true") or v == String("yes") or v == String("on"):
        return True
    return False
fn _is_falsy(s: String) -> Bool:
    var v = _lower_ascii(s).strip()
    if v == String("0") or v == String("false") or v == String("no") or v == String("off") or v == String(""):
        return True
    return False

# -------------------------------
# Safe integer parsing (Int64)
# -------------------------------
fn _parse_int64_safe(s: String, default: Int64) -> Int64:
    # Simple ASCII parse: optional leading +/- then digits
    var v = s.strip()
    var n = len(v)
    if n == 0:
        return default
    var i = 0
    var sign = Int64(1)
    var b0 = v.bytes()[0]
    if b0 == UInt8(45):          # '-'
        sign = Int64(-1)
        i = 1
    elif b0 == UInt8(43):        # '+'
        i = 1
    var acc = Int64(0)
    while i < n:
        var b = v.bytes()[i]
        if b < UInt8(48) or b > UInt8(57):
            return default
        var digit = Int64(b) - Int64(48)
        acc = acc * Int64(10) + digit
        i += 1
    return acc * sign

# -------------------------------

# -------------------------------
fn _parse_float64_safe(s: String, default: Float64) -> Float64:
    # Very conservative: only plain decimal with optional sign and one dot
    var v = s.strip()
    var n = len(v)
    if n == 0:
        return default
    var i = 0
    var sign = Float64(1.0)
    var b0 = v.bytes()[0]
    if b0 == UInt8(45):          # '-'
        sign = Float64(-1.0)
        i = 1
    elif b0 == UInt8(43):        # '+'
        i = 1
    var seen_dot = False
    var int_part = Int64(0)
    var frac_part = Int64(0)
    var frac_scale = Float64(1.0)
    while i < n:
        var b = v.bytes()[i]
        if b == UInt8(46):  # '.'
            if seen_dot:
                return default
            seen_dot = True
            i += 1
            continue
        if b < UInt8(48) or b > UInt8(57):
            return default
        if not seen_dot:
            int_part = int_part * Int64(10) + (Int64(b) - Int64(48))
        else:
            frac_part = frac_part * Int64(10) + (Int64(b) - Int64(48))
            frac_scale = frac_scale * Float64(10.0)
        i += 1
    var value = Float64(int_part) + Float64(frac_part) / frac_scale
    return sign * value

# -------------------------------
# Env: a simple immutable-map-like helper
# -------------------------------

struct Env:
    map: Dict[String, String]
fn __init__(out self) -> None:
        self.map = Dict[String, String]()
fn size(self) -> Int:
        return len(self.map)
fn keys(self) -> List[String]:
        var ks = List[String]()
        for k in self.map.keys():
            ks.append(k)
        return ks
fn has(self, key: String) -> Bool:
        return self.map.contains(key)
fn set(mut self, key: String, value: String) -> None:
        self.map[key] = value
fn get(self, key: String) -> String:
        if self.map.contains(key):
            return self.map[key]
        return String("")
fn get_or(self, key: String, default: String) -> String:
        if self.map.contains(key):
            return self.map[key]
        return default
fn get_bool(self, key: String, default: Bool) -> Bool:
        if not self.map.contains(key):
            return default
        var v = self.map[key]
        if _is_truthy(v):
            return True
        if _is_falsy(v):
            return False
        return default
fn get_i64(self, key: String, default: Int64) -> Int64:
        if not self.map.contains(key):
            return default
        return _parse_int64_safe(self.map[key], default)
fn get_f64(self, key: String, default: Float64) -> Float64:
        if not self.map.contains(key):
            return default
        return _parse_float64_safe(self.map[key], default)
fn merge(mut self, other: Env) -> None:
        # right-biased: other overrides self
        for k in other.map.keys():
            self.map[k] = other.map[k]
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -------------------------------
# Utilities
# -------------------------------
fn env_empty() -> Env:
    var e = Env()
    return e
fn env_single(key: String, value: String) -> Env:
    var e = Env()
    e.set(key, value)
    return e
fn env_from_pairs(keys: List[String], values: List[String]) -> Env:
    var e = Env()
    var n = len(keys)
    var m = len(values)
    var count = n if n < m else m
    var i = 0
    while i < count:
        e.set(keys[i], values[i])
        i += 1
    return e
fn copy_env(src: Env) -> Env:
    var e = Env()
    for k in src.map.keys():
        e.set(k, src.map[k])
    return e

# -------------------------------
# Minimal self-test hook (no prints)
# -------------------------------
fn _self_test() -> Bool:
    var ok = True
    var e = env_empty()
    e.set(String("A"), String("1"))
    e.set(String("B"), String("true"))
    e.set(String("C"), String("-42"))
    e.set(String("D"), String("3.14"))
    ok = ok and (e.size() == 4)
    ok = ok and e.get_or(String("X"), String("z")) == String("z")
    ok = ok and (e.get_i64(String("C"), Int64(0)) == Int64(-42))
    ok = ok and e.get_bool(String("B"), False)
    ok = ok and (_parse_float64_safe(String("2.5"), Float64(0.0)) > Float64(2.49))
    return ok