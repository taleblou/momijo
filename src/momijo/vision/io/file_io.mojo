# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/file_io.mojo
# Description: In-memory file store (no globals) + simple file I/O helpers
# Notes:
#  - English-only comments
#  - No global variables (state is explicit)
#  - Underscore helpers for other modules:
#       _read_file_bytes(path) -> (ok, bytes)
#       _write_file_bytes(path, data) -> ok
#       _write_file_raw(path, ptr, n) -> ok

from collections.list import List
from pathlib import Path
import os

# ----------------------------- String helpers -----------------------------

# Trim only leading/trailing ASCII spaces: ' ', '\t', '\n', '\r'
fn _trim_spaces(s: String) -> String:
    var n = len(s)
    if n == 0:
        return s

    fn _is_ws(u: String) -> Bool:
        return (u == String(" ")) or (u == String("\t")) or (u == String("\n")) or (u == String("\r"))

    var i = 0
    while i < n and _is_ws(String(s[i])):
        i += 1
    if i >= n:
        return String("")

    var j = n - 1
    while j >= i and _is_ws(String(s[j])):
        j -= 1

    var out = String("")
    var k = i
    while k <= j:
        out += String(s[k])
        k += 1
    return out

# Return last path component (works with '/' and '\'), ignoring trailing separators
fn _basename(p: String) -> String:
    var n = len(p)
    if n == 0:
        return p

    fn _is_sep(u: String) -> Bool:
        return (u == String("/")) or (u == String("\\"))

    var i = n - 1
    while i >= 0 and _is_sep(String(p[i])):
        i -= 1
    if i < 0:
        return String("")

    var j = i
    while j >= 0 and (not _is_sep(String(p[j]))):
        j -= 1

    var out = String("")
    var k = j + 1
    while k <= i:
        out += String(p[k])
        k += 1
    return out

# Parent directory (simple ASCII-only splitter)
fn _dirname(path: String) -> String:
    var i = path.__len__() - 1
    while i >= 0:
        var ch = path[i]
        if ch == '/':
            var j = 0
            var out = String()
            while j < i:
                out = out + String(path[j])
                j += 1
            return out.copy()
        i -= 1
    return ""   # current dir

# mkdir -p + probe write by open+close a temp file
fn _ensure_dir_writable(dirpath: String, DEBUG: Bool) -> Bool:
    if dirpath.__len__() == 0:
        # current directory
        try:
            var f = open(Path(".perm_test"), "w"); f.close()
            return True
        except _:
            if DEBUG: print("[MKDIR][ERR] current directory not writable")
            return False

    try:
        if DEBUG: print("[MKDIR] os.makedirs: " + dirpath)
        os.makedirs(dirpath)
    except e:
        if DEBUG: print("[MKDIR][warn] makedirs raised: " + String(e))

    var probe = dirpath + "/.perm_test"
    try:
        if DEBUG: print("[MKDIR] probing: " + probe)
        var f = open(Path(probe), "w"); f.close()
        if DEBUG: print("[MKDIR] probe ok (writable)")
        return True
    except _:
        if DEBUG: print("[MKDIR][ERR] cannot write to: " + dirpath)
        return False

# ----------------------------- In-Memory Store ------------------------------

struct MemFileStore(Copyable, Movable):
    var keys: List[String]
    var vals: List[List[UInt8]]

    fn __init__(out self):
        self.keys = List[String]()
        self.vals = List[List[UInt8]]()

    fn find(self, path: String) -> Int:
        var i = 0
        while i < self.keys.__len__():
            if self.keys[i] == path:
                return i
            i += 1
        return -1

    fn ensure(mut self, path: String) -> Int:
        var idx = self.find(path)
        if idx >= 0:
            return idx
        self.keys.append(path)
        self.vals.append(List[UInt8]())
        return self.keys.__len__() - 1

    fn write(mut self, path: String, data: List[UInt8]) -> Int:
        var idx = self.ensure(path)
        self.vals[idx] = data
        return self.vals[idx].__len__()

    fn append(mut self, path: String, raw_ptr: Pointer[UInt8], n: Int) -> Int:
        var idx = self.ensure(path)
        var i = 0
        while i < n:
            self.vals[idx].append(raw_ptr[i])
            i += 1
        return self.vals[idx].__len__()

    fn size(self, path: String) -> Int:
        var idx = self.find(path)
        if idx < 0:
            return 0
        return self.vals[idx].__len__()

    fn read(self, path: String) -> List[UInt8]:
        var idx = self.find(path)
        if idx < 0:
            return List[UInt8]()
        var src = self.vals[idx]
        var out = List[UInt8]()
        var i = 0
        while i < src.__len__():
            out.append(src[i])
            i += 1
        return out.copy()

    fn remove(mut self, path: String) -> Bool:
        var idx = self.find(path)
        if idx < 0:
            return False
        var last = self.keys.__len__() - 1
        self.keys[idx] = self.keys[last]
        self.vals[idx] = self.vals[last]
        self.keys.pop()
        self.vals.pop()
        return True

    fn list(self) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < self.keys.__len__():
            out.append(self.keys[i])
            i += 1
        return out.copy()

    fn clear(mut self) -> Int:
        var count = self.keys.__len__()
        while self.keys.__len__() > 0:
            self.keys.pop()
        while self.vals.__len__() > 0:
            self.vals.pop()
        return count

# ----------------------------- Small helpers -----------------------------

fn _dummy_ptr() -> Pointer[UInt8]:
    return Pointer[UInt8].null()

fn _empty_bytes() -> List[UInt8]:
    return List[UInt8]()

fn _empty_names() -> List[String]:
    return List[String]()

# ----------------------------- Legacy-shaped API ---------------------------
# (Bool ok, Int size, List[UInt8] bytes, List[String] names, Pointer[UInt8] dummy)

fn memfs_try_read(fs: MemFileStore, path: String) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var data = fs.read(path)
    var ok = data.__len__() > 0 or (fs.find(path) >= 0)
    return (ok, data.__len__(), data, _empty_names(), _dummy_ptr())

fn memfs_write_replace(mut fs: MemFileStore, path: String, data: List[UInt8]) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var sz = fs.write(path, data)
    return (True, sz, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_append_ptr(mut fs: MemFileStore, path: String, raw_ptr: Pointer[UInt8], n: Int) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var sz = fs.append(path, raw_ptr, n)
    return (True, sz, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_size_of(fs: MemFileStore, path: String) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var sz = fs.size(path)
    var ok = sz > 0 or (fs.find(path) >= 0)
    return (ok, sz, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_remove_path(mut fs: MemFileStore, path: String) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var ok = fs.remove(path)
    return (ok, 0, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_list_all(fs: MemFileStore) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var names = fs.list()
    return (True, names.__len__(), _empty_bytes(), names, _dummy_ptr())

fn memfs_clear_all(mut fs: MemFileStore) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var removed = fs.clear()
    return (True, removed, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_has_path(fs: MemFileStore, path: String) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var exists = fs.find(path) >= 0
    return (exists, 0, _empty_bytes(), _empty_names(), _dummy_ptr())

fn memfs_ensure_path(mut fs: MemFileStore, path: String) -> (Bool, Int, List[UInt8], List[String], Pointer[UInt8]):
    var idx = fs.ensure(path)
    var ok = idx >= 0
    return (ok, 0, _empty_bytes(), _empty_names(), _dummy_ptr())

# ----------------------------- On-disk I/O (underscored) --------------------
# Single point of truth for disk I/O.
# IMPORTANT:
#   - Use read_bytes()/write_bytes() for binary-safe I/O.
#   - Modes: "r" (read), "w" (write), "rw" (read-write). No separate "b".

fn _read_file_bytes(path: String) -> (Bool, List[UInt8]):
    try:
        var f = open(Path(path), "r")
        try:
            var bytes = f.read_bytes()
            f.close()
            return (True, bytes.copy())
        except _:
            try: f.close() except _: pass
            return (False, List[UInt8]())
    except _:
        return (False, List[UInt8]())

# Write whole buffer to disk (binary-safe)
fn _write_file_bytes(path: String, data: List[UInt8]) -> Bool:
    var n = data.__len__()
    try:
        var f = open(Path(path), "w")
        try:
            if n > 0:
                var off = 0
                # write in chunks to limit stack/heap bursts
                var CHUNK = 65536
                var buf = UnsafePointer[UInt8].alloc(CHUNK)
                while off < n:
                    var chunk = n - off
                    if chunk > CHUNK: chunk = CHUNK
                    var i = 0
                    while i < chunk:
                        buf[i] = data[off + i]
                        i += 1
                    var span = Span[UInt8](buf, chunk)
                    f.write_bytes(span)
                    off += chunk
                UnsafePointer[UInt8].free(buf)
            f.close()
            return True
        except _:
            try: f.close() except _: pass
            return False
    except _:
        return False

# Overload for UnsafePointer[UInt8] sources (e.g., JPEG/PNG encoders)
fn _write_file_raw(path: String, raw_ptr: UnsafePointer[UInt8], n: Int) -> Bool:
    var p = _trim_spaces(path)
    if len(p) == 0:
        print("[_write_file_raw] FAIL: empty path after trim")
        return False

    try:
        var f = open(Path(p), "w")
        try:
            if n > 0:
                var off = 0
                var CHUNK = 65536
                var buf = UnsafePointer[UInt8].alloc(CHUNK)
                while off < n:
                    var chunk = n - off
                    if chunk > CHUNK: chunk = CHUNK
                    var j = off
                    var i = 0
                    while i < chunk:
                        buf[i] = raw_ptr[j]
                        i += 1
                        j += 1
                    var span = Span[UInt8](buf, chunk)
                    f.write_bytes(span)
                    off += chunk
                UnsafePointer[UInt8].free(buf)
            f.close()
            return True
        except _:
            try: f.close() except _: pass
            return False
    except _:
        return False

fn read_all_bytes(path: String) raises -> List[UInt8]:
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(65536)
            if len(chunk) == 0: break
            out.extend(chunk.copy())
    return out.copy()

@always_inline
fn _nz_u8(buf: List[UInt8]) -> Int:
    var c = 0; var i = 0
    while i < len(buf):
        if buf[i] != UInt8(0): c += 1
        i += 1
    return c

@always_inline
fn _sum_u8(buf: List[UInt8]) -> Int:
    var s = 0; var i = 0
    while i < len(buf):
        s += Int(buf[i]); i += 1
    return s
fn _count_nz(a: List[UInt8]) -> Int:
    var c = 0
    var i = 0
    while i < len(a):
        if a[i] != UInt8(0):
            c += 1
        i += 1
    return c

fn write_all_bytes(path: String, data: List[UInt8]) raises -> Bool:
    with open(path, "w") as f:     # فقط "w" مجازه
        var view = Span(data)
        var n = f.write_bytes(view)
        # print("[io] write →", path, " bytes=", " sum=", _sum_u8(data), " nz=", _count_nz(data))

    # دیباگ امضای PNG
    # if len(data) >= 8:
        # print("[io] png-signature=", data[0], ",", data[1], ",", data[2], ",", data[3], ",", data[4], ",", data[5], ",", data[6], ",", data[7])
    # print("[io] done writing to ", path)
    return True
