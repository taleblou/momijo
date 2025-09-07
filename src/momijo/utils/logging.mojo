# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/logging.mojo

from stdlib.list import List
from stdlib.string import String

fn _lvl_debug() -> UInt8: return UInt8(10)
fn _lvl_info()  -> UInt8: return UInt8(20)
fn _lvl_warn()  -> UInt8: return UInt8(30)
fn _lvl_error() -> UInt8: return UInt8(40)
fn _lvl_off()   -> UInt8: return UInt8(255)
fn level_name(lvl: UInt8) -> String:
    if lvl == _lvl_debug(): return String("DEBUG")
    if lvl == _lvl_info():  return String("INFO")
    if lvl == _lvl_warn():  return String("WARN")
    if lvl == _lvl_error(): return String("ERROR")
    if lvl == _lvl_off():   return String("OFF")
    return String("LVL(") + String(String(Int64(lvl))) + String(")")
fn parse_level(s: String, default: UInt8 = _lvl_info()) -> UInt8:
    var t = s.upper().strip()
    if t == String("DEBUG"): return _lvl_debug()
    if t == String("INFO"):  return _lvl_info()
    if t == String("WARN") or t == String("WARNING"): return _lvl_warn()
    if t == String("ERROR") or t == String("ERR"): return _lvl_error()
    if t == String("OFF"):   return _lvl_off()
    # accept numeric string 0..255
    var n = len(t)
    if n > 0:
        var i = 0
        var ok = True
        while i < n:
            var b = t.bytes()[i]
            if b < UInt8(48) or b > UInt8(57):
                ok = False
                break
            i += 1
        if ok:
            var acc = Int64(0)
            i = 0
            while i < n:
                acc = acc * Int64(10) + (Int64(t.bytes()[i]) - Int64(48))
                i += 1
            if acc < Int64(0): acc = Int64(0)
            if acc > Int64(255): acc = Int64(255)
            return UInt8(acc)
    return default

# -------------------------------
# String helpers
# -------------------------------
fn _sanitize_line(s: String) -> String:
    # Replace CR/LF with spaces to keep single-line logs deterministic
    var out = String("")
    var i = 0
    var n = len(s)
    while i < n:
        var b = s.bytes()[i]
        if b == UInt8(10) or b == UInt8(13):
            out = out + String(" ")
        else:
            out = out + String.from_utf8([b])
        i += 1
    return out
fn _join_kv(keys: List[String], vals: List[String]) -> String:
    var buf = String("")
    var n = len(keys)
    var m = len(vals)
    var c = n if n < m else m
    var i = 0
    while i < c:
        if i > 0: buf = buf + String(" ")
        buf = buf + _sanitize_line(keys[i]) + String("=") + _sanitize_line(vals[i])
        i += 1
    return buf

# -------------------------------
# Logger
# -------------------------------
struct Logger:
    level: UInt8           # minimum level to emit
    name: String           # optional component/service name
    show_level: Bool
    show_name: Bool
    show_time: Bool        # placeholder flag; timestamp injection via arg
    sep: String            # separator between header and message
fn __init__(out self,
                level: UInt8 = _lvl_info(),
                name: String = String(""),
                show_level: Bool = True,
                show_name: Bool = False,
                show_time: Bool = False,
                sep: String = String(" | ")):
        self.level = level
        self.name = name
        self.show_level = show_level
        self.show_name = show_name
        self.show_time = show_time
        self.sep = sep
fn set_level(mut self, lvl: UInt8) -> None:
        self.level = lvl
fn enabled(self, msg_level: UInt8) -> Bool:
        # emit when message level is >= current level, unless OFF
        if self.level == _lvl_off(): return False
        return Int64(msg_level) >= Int64(self.level)
fn header(self, msg_level: UInt8, ts: String) -> String:
        var h = String("")
        if self.show_level:
            h = h + level_name(msg_level)
        if self.show_name and len(self.name) > 0:
            if len(h) > 0: h = h + String(" ")
            h = h + String("[") + self.name + String("]")
        if self.show_time and len(ts) > 0:
            if len(h) > 0: h = h + String(" ")
            h = h + ts
        return h
fn format(self, msg_level: UInt8, msg: String, ts: String = String(""), kv_keys: List[String] = List[String](), kv_vals: List[String] = List[String]()) -> String:
        var head = self.header(msg_level, ts)
        var body = _sanitize_line(msg)
        var out = String("")
        if len(head) > 0:
            out = head + self.sep + body
        else:
            out = body
        # append key-values if present
        var kvs = _join_kv(kv_keys, kv_vals)
        if len(kvs) > 0:
            out = out + String(" -- ") + kvs
        return out
fn emit(self, msg_level: UInt8, msg: String, ts: String = String(""), kv_keys: List[String] = List[String](), kv_vals: List[String] = List[String]()):
        if not self.enabled(msg_level): return
        var line = self.format(msg_level, msg, ts, kv_keys, kv_vals)
        # printing via String(...)
        print(String(line))

    # Convenience level-specific methods
fn debug(self, msg: String, ts: String = String("")):
        self.emit(_lvl_debug(), msg, ts)
fn info(self, msg: String, ts: String = String("")):
        self.emit(_lvl_info(), msg, ts)
fn warn(self, msg: String, ts: String = String("")):
        self.emit(_lvl_warn(), msg, ts)
fn error(self, msg: String, ts: String = String("")):
        self.emit(_lvl_error(), msg, ts)

    # With key-value pairs
fn debug_kv(self, msg: String, keys: List[String], vals: List[String], ts: String = String("")):
        self.emit(_lvl_debug(), msg, ts, keys, vals)
fn info_kv(self, msg: String, keys: List[String], vals: List[String], ts: String = String("")):
        self.emit(_lvl_info(), msg, ts, keys, vals)
fn warn_kv(self, msg: String, keys: List[String], vals: List[String], ts: String = String("")):
        self.emit(_lvl_warn(), msg, ts, keys, vals)
fn error_kv(self, msg: String, keys: List[String], vals: List[String], ts: String = String("")):
        self.emit(_lvl_error(), msg, ts, keys, vals)

    # Derive a child logger (e.g., sub-component) without mutating the parent
fn child(self, suffix: String) -> Logger:
        var nm = self.name
        if len(nm) > 0:
            nm = nm + String(".") + suffix
        else:
            nm = suffix
        var l = Logger(self.level, nm, self.show_level, True, self.show_time, self.sep)
        return l
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -------------------------------
# Factories
# -------------------------------
fn logger_default() -> Logger:
    var l = Logger(_lvl_info(), String(""), True, False, False, String(" | "))
    return l
fn logger_with_level(level: UInt8) -> Logger:
    var l = Logger(level, String(""), True, False, False, String(" | "))
    return l
fn logger_named(name: String, level: UInt8 = _lvl_info()) -> Logger:
    var l = Logger(level, name, True, True, False, String(" | "))
    return l
fn logger_from_level_str(level_str: String, name: String = String("")) -> Logger:
    var lvl = parse_level(level_str, _lvl_info())
    var l = Logger(lvl, name, True, len(name) > 0, False, String(" | "))
    return l

# -------------------------------
# Minimal self-test
# -------------------------------
fn _self_test() -> Bool:
    var ok = True
    var l = logger_named(String("core"), _lvl_debug())
    # should be enabled for all >= DEBUG
    ok = ok and l.enabled(_lvl_debug())
    ok = ok and l.enabled(_lvl_info())
    ok = ok and l.enabled(_lvl_warn())
    ok = ok and l.enabled(_lvl_error())

    var msg = l.format(_lvl_info(), String("hello"))
    ok = ok and (len(msg) > 0)

    var keys = List[String](); keys.append(String("user")); keys.append(String("id"))
    var vals = List[String](); vals.append(String("mitra")); vals.append(String("42"))
    var with_kv = l.format(_lvl_info(), String("hi"), String("2025-09-05T12:00:00Z"), keys, vals)
    ok = ok and (len(with_kv) > len(msg))

    var off = Logger(_lvl_off(), String(""), True, False, False, String(" | "))
    ok = ok and (not off.enabled(_lvl_error()))

    return ok