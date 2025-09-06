# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.checkpoint
# Path:   src/momijo/nn/checkpoint.mojo
#
# A tiny, dependency-light checkpoint utility for educational models.
# Focus: in-memory packing/unpacking of parameter-like arrays (no file I/O).
# Supported array ranks: 1D, 2D, 3D, 4D (Float64 lists).
#
# Main ideas
# ----------
# - Build a `Checkpoint`, add named arrays via `add_*` helpers.
# - Apply a checkpoint into existing buffers with `apply_*_into(name, dst)`.
# - Export a human-readable summary string with `export_text()`.
# - No external dependencies or global state. No serialization to disk;
#   you can use `export_text()` if you want a quick inspection/log.
#
# Shapes & Conventions
# --------------------
# - For 4D, we assume NCHW ordering (N, C, H, W) when describing sizes.
# - This module uses copies (immutability-ish behavior) to keep things simple.
# - If target/destination shapes mismatch during `apply_*_into`, we copy the
#   overlapping region and ignore the rest (safe and predictable).

# --------- Utilities ---------
fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x

fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn _zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

fn _zeros3d(ch: Int, h: Int, w: Int) -> List[List[List[Float64]]]:
    var y = List[List[List[Float64]]]()
    for c in range(ch):
        y.push(_zeros2d(h, w))
    return y

fn _zeros4d(n: Int, ch: Int, h: Int, w: Int) -> List[List[List[List[Float64]]]]:
    var y = List[List[List[List[Float64]]]]()
    for i in range(n):
        y.push(_zeros3d(ch, h, w))
    return y

fn _copy1d(x: List[Float64]) -> List[Float64]:
    var n = len(x)
    var y = List[Float64]()
    for i in range(n): y.push(x[i])
    return y

fn _copy2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var r = len(x)
    var y = List[List[Float64]]()
    for i in range(r):
        var c = 0
        if len(x[i]) > 0: c = len(x[i])
        var row = List[Float64]()
        for j in range(c): row.push(x[i][j])
        y.push(row)
    return y

fn _copy3d(x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
    var c = len(x)
    var y = List[List[List[Float64]]]()
    for i in range(c):
        y.push(_copy2d(x[i]))
    return y

fn _copy4d(x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
    var n = len(x)
    var y = List[List[List[List[Float64]]]]()
    for i in range(n):
        y.push(_copy3d(x[i]))
    return y

# --------- Named arrays ---------
struct A1:
    var name: String
    var data: List[Float64]

struct A2:
    var name: String
    var data: List[List[Float64]]

struct A3:
    var name: String
    var data: List[List[List[Float64]]]

struct A4:
    var name: String
    var data: List[List[List[List[Float64]]]]

struct KV:
    var key: String
    var value: String

# --------- Checkpoint container ---------
struct Checkpoint:
    var version: Int
    var meta: List[KV]
    var a1: List[A1]
    var a2: List[A2]
    var a3: List[A3]
    var a4: List[A4]

    fn __init__(out self):
        self.version = 1
        self.meta = List[KV]()
        self.a1 = List[A1]()
        self.a2 = List[A2]()
        self.a3 = List[A3]()
        self.a4 = List[A4]()

    # ---- meta ----
    fn add_meta(mut self, key: String, value: String):
        var kv = KV(key, value)
        self.meta.push(kv)

    # ---- add arrays (copying) ----
    fn add_1d(mut self, name: String, data: List[Float64]):
        var x = _copy1d(data)
        var e = A1(name, x)
        self.a1.push(e)

    fn add_2d(mut self, name: String, data: List[List[Float64]]):
        var x = _copy2d(data)
        var e = A2(name, x)
        self.a2.push(e)

    fn add_3d(mut self, name: String, data: List[List[List[Float64]]]):
        var x = _copy3d(data)
        var e = A3(name, x)
        self.a3.push(e)

    fn add_4d(mut self, name: String, data: List[List[List[List[Float64]]]]):
        var x = _copy4d(data)
        var e = A4(name, x)
        self.a4.push(e)

    # ---- getters ----
    fn has_1d(self, name: String) -> Bool:
        for i in range(len(self.a1)):
            if self.a1[i].name == name: return True
        return False

    fn has_2d(self, name: String) -> Bool:
        for i in range(len(self.a2)):
            if self.a2[i].name == name: return True
        return False

    fn has_3d(self, name: String) -> Bool:
        for i in range(len(self.a3)):
            if self.a3[i].name == name: return True
        return False

    fn has_4d(self, name: String) -> Bool:
        for i in range(len(self.a4)):
            if self.a4[i].name == name: return True
        return False

    # ---- apply/copy into provided buffers (shape-safe, overlap copy) ----
    fn apply_1d_into(self, name: String, mut dst: List[Float64]) -> Int:
        for i in range(len(self.a1)):
            if self.a1[i].name == name:
                var n = _min(len(self.a1[i].data), len(dst))
                for k in range(n): dst[k] = self.a1[i].data[k]
                return n
        return 0

    fn apply_2d_into(self, name: String, mut dst: List[List[Float64]]) -> (Int, Int):
        for i in range(len(self.a2)):
            if self.a2[i].name == name:
                var r = _min(len(self.a2[i].data), len(dst))
                var c = 0
                if r > 0:
                    c = _min(len(self.a2[i].data[0]), len(dst[0]))
                for rr in range(r):
                    for cc in range(c):
                        dst[rr][cc] = self.a2[i].data[rr][cc]
                return (r, c)
        return (0, 0)

    fn apply_3d_into(self, name: String, mut dst: List[List[List[Float64]]]) -> (Int, Int, Int):
        for i in range(len(self.a3)):
            if self.a3[i].name == name:
                var C = _min(len(self.a3[i].data), len(dst))
                var H = 0; var W = 0
                if C > 0:
                    H = _min(len(self.a3[i].data[0]), len(dst[0]))
                    if H > 0: W = _min(len(self.a3[i].data[0][0]), len(dst[0][0]))
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            dst[c][h][w] = self.a3[i].data[c][h][w]
                return (C, H, W)
        return (0, 0, 0)

    fn apply_4d_into(self, name: String, mut dst: List[List[List[List[Float64]]]]) -> (Int, Int, Int, Int):
        for i in range(len(self.a4)):
            if self.a4[i].name == name:
                var N = _min(len(self.a4[i].data), len(dst))
                var C = 0; var H = 0; var W = 0
                if N > 0:
                    C = _min(len(self.a4[i].data[0]), len(dst[0]))
                    if C > 0:
                        H = _min(len(self.a4[i].data[0][0]), len(dst[0][0]))
                        if H > 0:
                            W = _min(len(self.a4[i].data[0][0][0]), len(dst[0][0][0]))
                for n in range(N):
                    for c in range(C):
                        for h in range(H):
                            for w in range(W):
                                dst[n][c][h][w] = self.a4[i].data[n][c][h][w]
                return (N, C, H, W)
        return (0, 0, 0, 0)

    # ---- Export a human-readable summary ----
    fn export_text(self) -> String:
        var s = String("MOMIJO_CHECKPOINT v=") + String(self.version)
        s = s + String("\nMETA:") 
        for i in range(len(self.meta)):
            s = s + String(" ") + self.meta[i].key + String("=") + self.meta[i].value
        s = s + String("\nA1:")
        for i in range(len(self.a1)):
            s = s + String(" ") + self.a1[i].name + String("[") + String(len(self.a1[i].data)) + String("]")
        s = s + String("\nA2:")
        for i in range(len(self.a2)):
            var r = len(self.a2[i].data)
            var c = 0
            if r > 0: c = len(self.a2[i].data[0])
            s = s + String(" ") + self.a2[i].name + String("[") + String(r) + String(",") + String(c) + String("]")
        s = s + String("\nA3:")
        for i in range(len(self.a3)):
            var C = len(self.a3[i].data)
            var H = 0; var W = 0
            if C > 0:
                H = len(self.a3[i].data[0])
                if H > 0: W = len(self.a3[i].data[0][0])
            s = s + String(" ") + self.a3[i].name + String("[") + String(C) + String(",") + String(H) + String(",") + String(W) + String("]")
        s = s + String("\nA4:")
        for i in range(len(self.a4)):
            var N = len(self.a4[i].data)
            var C = 0; var H = 0; var W = 0
            if N > 0:
                C = len(self.a4[i].data[0])
                if C > 0:
                    H = len(self.a4[i].data[0][0])
                    if H > 0: W = len(self.a4[i].data[0][0][0])
            s = s + String(" ") + self.a4[i].name + String("[") + String(N) + String(",") + String(C) + String(",") + String(H) + String(",") + String(W) + String("]")
        return s

# --------- Convenience pack/apply helpers ---------
fn pack_linear(name_w: String, name_b: String, W: List[List[Float64]], b: List[Float64]) -> Checkpoint:
    var ck = Checkpoint()
    ck.add_2d(name_w, W)
    ck.add_1d(name_b, b)
    return ck

# Safe apply: copy from ckpt into provided buffers if present
fn apply_linear(mut ck: Checkpoint, name_w: String, name_b: String, mut W: List[List[Float64]], mut b: List[Float64]) -> (Int, Int):
    var (r, c) = ck.apply_2d_into(name_w, W)
    var n = ck.apply_1d_into(name_b, b)
    return (r * c, n)

# --------- Tests & equality checks ---------
fn _eq1d(a: List[Float64], b: List[Float64], tol: Float64 = 1e-9) -> Bool:
    var n = len(a)
    if n != len(b): return False
    for i in range(n):
        if _abs(a[i] - b[i]) > tol: return False
    return True

fn _eq2d(a: List[List[Float64]], b: List[List[Float64]], tol: Float64 = 1e-9) -> Bool:
    var r = len(a)
    if r != len(b): return False
    var c = 0
    if r > 0: c = len(a[0])
    if r > 0 and c != len(b[0]): return False
    for i in range(r):
        for j in range(c):
            if _abs(a[i][j] - b[i][j]) > tol: return False
    return True

# --------- Smoke test ---------
fn _self_test() -> Bool:
    var ok = True

    # Create some fake params
    var W = _zeros2d(2, 3)
    W[0][0] = 0.1; W[0][1] = 0.2; W[0][2] = -0.3
    W[1][0] = -0.4; W[1][1] = 0.5; W[1][2] = 0.6
    var b = _zeros1d(2)
    b[0] = 0.01; b[1] = -0.02

    # Pack
    var ck = pack_linear(String("linear.W"), String("linear.b"), W, b)
    ck.add_meta(String("arch"), String("Linear(3->2)"))
    ck.add_meta(String("notes"), String("toy example"))

    # Export summary
    var txt = ck.export_text()
    ok = ok and (len(txt) > 0)

    # Apply into fresh buffers
    var W2 = _zeros2d(2, 3)
    var b2 = _zeros1d(2)
    var (_cntW, _cntb) = apply_linear(ck, String("linear.W"), String("linear.b"), W2, b2)
    ok = ok and _eq2d(W, W2) and _eq1d(b, b2)

    # Partial-shape application (mismatch)
    var W3 = _zeros2d(1, 2)
    var b3 = _zeros1d(1)
    var (_cntW3, _cntb3) = apply_linear(ck, String("linear.W"), String("linear.b"), W3, b3)
    ok = ok and (W3[0][0] == 0.1) and (W3[0][1] == 0.2) and (b3[0] == 0.01)

    return ok

 
