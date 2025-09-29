# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/file_io.mojo
# Description: In-memory file store (no globals) + simple file I/O helpers
# Notes:
#  - English-only comments
#  - No global variables (state is explicit)
#  - Provide underscore helpers for other modules:
#       _read_file_bytes(path) -> (ok, bytes)
#       _write_file_bytes(path, data) -> ok
#       _write_file_raw(path, ptr, n) -> ok

from collections.list import List
from pathlib import Path
import os
 

fn _basename(path: String) -> String:
    var i = path.__len__() - 1
    while i >= 0:
        var ch = path[i]
        if ch == '/':
            var j = i + 1
            var out = String()
            while j < path.__len__():
                out = out + String(path[j])
                j += 1
            return out
        i -= 1
    return path

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
            return out
        i -= 1
    return ""   # current dir

# mkdir -p + probe write by open+close a temp file
fn _ensure_dir_writable(dirpath: String, DEBUG: Bool) -> Bool:
    if dirpath.__len__() == 0:
        # current directory
        try:
            var f = open(".perm_test", "wb"); f.close()
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
        var f = open(probe, "wb"); f.close()
        if DEBUG: print("[MKDIR] probe ok (writable)")
        return True
    except _:
        if DEBUG: print("[MKDIR][ERR] cannot write to: " + dirpath)
        return False
# ----------------------------- In-Memory Store ------------------------------

struct MemFileStore(ExplicitlyCopyable, Movable):
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
        return out

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
        return out

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

fn _read_file_bytes(path: String) -> (Bool, List[UInt8]):
    var out = List[UInt8]()
    try:
        var f = open(path, "rb")
        try:
            while True:
                var chunk = f.read(4096)  # String (bytes chunk)
                var m = chunk.__len__()
                if m == 0:
                    break
                var i = 0
                while i < m:
                    var ch = chunk[i]     # Char
                    out.append(UInt8(Int(ch)))
                    i += 1
            f.close()
            return (True, out)
        except _:
            try: f.close() except _: pass
            return (False, List[UInt8]())
    except _:
        return (False, List[UInt8]())
    # Compiler requires a final explicit return on all paths:
    return (False, List[UInt8]())

# Write to disk with multi-step fallback: target → /tmp → .
fn _write_file_bytes(path: String, data: List[UInt8]) -> Bool:
    var DEBUG = True
    var n = data.__len__()
    if DEBUG:
        print("[WRITE] path=" + path)
        print("[WRITE] data.len=" + String(n))

    try:
        var f = open(Path(path), "w")   # supported modes: r, w, rw
        try:
            if n > 0:
                var CHUNK = 65536
                var buf = UnsafePointer[UInt8].alloc(CHUNK)
                var off = 0
                while off < n:
                    var chunk = n - off
                    if chunk > CHUNK: chunk = CHUNK
                    var i = 0
                    while i < chunk:
                        buf[i] = data[off + i]
                        i += 1
                    # construct a Span over the temp buffer and write as bytes
                    var span = Span[UInt8](buf, chunk)
                    f.write_bytes(span)  # FileHandle API
                    if DEBUG:
                        print("[WRITE] off=" + String(off) + " chunk=" + String(chunk))
                    off += chunk
                UnsafePointer[UInt8].free(buf)
            f.close()
            if DEBUG: print("[WRITE] closed ok")
            return True
        except _:
            if DEBUG: print("[WRITE][ERR] exception during write loop")
            try: f.close() except _: pass
            return False
    except e:
        if DEBUG: print("[WRITE][EXC] open(Path(path), 'w') failed: " + String(e))
        return False

    return False



# Overload to support UnsafePointer[UInt8] callers (e.g., JPEG encoder)
fn _write_file_raw(path: String, raw_ptr: UnsafePointer[UInt8], n: Int) -> Bool:
    if n <= 0:
        try:
            var f0 = open(path, "wb")
            try:
                f0.close()
                return True
            except _:
                try: f0.close() except _: pass
                return False
        except _:
            return False

    var buf = UnsafePointer[UInt8].alloc(4096)
    var ok = False
    try:
        var f = open(path, "wb")
        try:
            var off = 0
            while off < n:
                var chunk = n - off
                if chunk > 4096:
                    chunk = 4096
                var j = off
                var i = 0
                while i < chunk:
                    buf[i] = raw_ptr[j]
                    i += 1
                    j += 1
                _ = f.write(buf, chunk)
                off += chunk
            f.close()
            ok = True
        except _:
            try: f.close() except _: pass
            ok = False
    except _:
        ok = False
    UnsafePointer[UInt8].free(buf)
    return ok
