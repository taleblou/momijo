# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.profiler.core
# File:         src/momijo/learn/profiler/core.mojo
#
# Description:
#   Minimal CPU-only profiler core with a simple row aggregator and table printer.

from collections.list import List

struct ProfRow(Copyable, Movable):
    var key: String
    var calls: Int
    var est_ops: Float64
    var shapes: String

    fn __init__(out self, key: String, calls: Int, est_ops: Float64, shapes: String):
        self.key = key
        self.calls = calls
        self.est_ops = est_ops
        self.shapes = shapes

    fn __copyinit__(out self, other: Self):
        self.key = other.key
        self.calls = other.calls
        self.est_ops = other.est_ops
        self.shapes = other.shapes

struct Profiler(Copyable, Movable):
    var rows: List[ProfRow]
    var record_shapes: Bool

    fn __init__(out self, record_shapes: Bool = True):
        self.rows = List[ProfRow]()
        self.record_shapes = record_shapes

    fn __copyinit__(out self, other: Self):
        # shallow row copy (values)
        self.rows = other.rows.copy()
        self.record_shapes = other.record_shapes 

    fn record(mut self, key: String, est_ops: Float64, shapes: String):
        var i = 0
        while i < len(self.rows):
            if self.rows[i].key == key:
                var r = self.rows[i].copy()
                r.calls = r.calls + 1
                r.est_ops = r.est_ops + est_ops
                if self.record_shapes and len(r.shapes) == 0 and len(shapes) > 0:
                    r.shapes = shapes
                self.rows[i] = r.copy()
                return
            i = i + 1
        self.rows.append(ProfRow(key, 1, est_ops, shapes if self.record_shapes else ""))

    fn table(self, sort_by_ops: Bool = True, row_limit: Int = 5) -> String:
        # copy rows
        var rs = List[ProfRow]()
        var i = 0
        while i < len(self.rows):
            rs.append(self.rows[i].copy())
            i = i + 1

        # sort by est_ops desc (selection sort)
        if sort_by_ops:
            var n = len(rs)
            var a = 0
            while a < n:
                var max_i = a
                var b = a + 1
                while b < n:    
                    if rs[b].est_ops > rs[max_i].est_ops:
                        max_i = b
                    b = b + 1
                if max_i != a:
                    var tmp = rs[a].copy(); rs[a] = rs[max_i].copy(); rs[max_i] = tmp.copy()
                a = a + 1

        # format
        var lines = List[String]()
        lines.append("Key                        Calls   EstOps (×1e6)   Shapes")
        lines.append("--------------------------------------------------------------")

        var limit = row_limit
        var j = 0
        while j < len(rs) and j < limit:
            var r = rs[j].copy()
            var est_m = r.est_ops / 1_000_000.0
            var calls_s = String(r.calls)
            var est_s = String(est_m)

            var pad1 = 26 - len(r.key)
            if pad1 < 1: pad1 = 1
            var pad2 = 7 - len(calls_s)
            if pad2 < 1: pad2 = 1
            var pad3 = 15 - len(est_s)
            if pad3 < 1: pad3 = 1

            var line = r.key
            line = line + _spaces(pad1)
            line = line + calls_s
            line = line + _spaces(pad2)
            line = line + est_s
            line = line + _spaces(pad3)
            line = line + r.shapes
            lines.append(line)

            j = j + 1

        return "\n".join(lines)


@always_inline
fn _spaces(n: Int) -> String:
    var n2 = n
    if n2 < 0: n2 = 0
    var out = ""
    var i = 0
    while i < n2:
        out = out + " "
        i = i + 1
    return out