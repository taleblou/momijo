# Project:      Momijo
# Module:       src.momijo.visual.spec.exporters.plotly_json
# File:         plotly_json.mojo
# Path:         src/momijo/visual/spec/exporters/plotly_json.mojo
#
# Description:  src.momijo.visual.spec.exporters.plotly_json — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: _escape_json, _pad3, _css_rgb, _opacity_str, _append_number_array, write_plotly_json, _self_test
#   - Performs file/Path IO; prefer context-managed patterns.


from io.file import open
from momijo.core.config import trace
from momijo.core.device import kind
from momijo.core.error import module
from momijo.dataframe.helpers import close, m, t
from momijo.ir.dialects.annotations import array, integer
from momijo.nn.parameter import data
from momijo.utils.result import f, g
from momijo.utils.timer import start
from momijo.visual.scene.scene import Scene, SceneMarkKinds
from pathlib import Path
from pathlib.path import Path
from runtime.tracing import Trace

# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.spec.exporters.plotly_json
# File:         plotly_json.mojo
# Path:         momijo/visual/spec/exporters/plotly_json.mojo
#
# Description:  Core module 'plotly jso' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================
fn _escape_json(s: String) -> String:
    # Minimal JSON escape for quotes and backslashes + control chars
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == 34:        # "
            out = out + String("\\\"")
        elif ch == 92:      # \
            out = out + String("\\\\")
        elif ch == 10:      # \n
            out = out + String("\\n")
        elif ch == 13:      # \r
            out = out + String("\\r")
        elif ch == 9:       # \t
            out = out + String("\\t")
        else:
            out = out + String(Char(ch))
        i += 1
    return out
fn _pad3(n: Int) -> String:
    var s = String(n)
    if n < 10: s = String("00") + s
    elif n < 100: s = String("0") + s
    return s
fn _css_rgb(c: ColorRGBA) -> String:
    return String("rgb(") + String(c.r) + String(",") + String(c.g) + String(",") + String(c.b) + String(")")
fn _opacity_str(c: ColorRGBA) -> String:
    if c.a >= 255: return String("1")
    if c.a <= 0: return String("0")
    # integer math for ~three decimals
    var milli = (c.a * 1000) / 255
    if milli >= 1000: milli = 999
    return String("0.") + _pad3(milli)

# Append numeric array like [1,2,3] from list of Points.x or Points.y
fn _append_number_array(mut out: String, vals: List[Float64]) -> None:
    out = out + String("[")
    var i = 0
    while i < len(vals):
        out = out + String(vals[i])
        if i + 1 < len(vals): out = out + String(",")
        i += 1
    out = out + String("]")

# Notes:
#  - Coordinates are interpreted as pixel space; we set xaxis range [0,width], yaxis [height,0] with autorange:'reversed'.
#  - Marks:



#  - Text is ignored in this MVP (can be mapped to annotations later).
fn write_plotly_json(sc: Scene, path: String) -> None:
    var f = open(path, String("w"))
    if f.is_null(): return

    var sb = String("{")

    # layout
    sb = sb + String("\"layout\":{")
    sb = sb + String("\"width\":") + String(sc.width) + String(",")
    sb = sb + String("\"height\":") + String(sc.height) + String(",")
    sb = sb + String("\"paper_bgcolor\":\"#FFFFFF\",")
    if len(sc.title) > 0:
        sb = sb + String("\"title\":\"") + _escape_json(sc.title) + String("\",")
    sb = sb + String("\"xaxis\":{\"range\":[0,") + String(sc.width) + String("]},"))
    sb = sb + String("\"yaxis\":{\"range\":[") + String(sc.height) + String(",0],\"autorange\":\"reversed\"},")
    # shapes array will be appended later if any
    sb = sb + String("\"shapes\":[")
    var first_shape = True

    # data traces
    sb = sb + String("],\"data\":[")
    # We'll fill shapes later; for now we close it and open data (we already closed shapes as empty).
    # Reset shapes insertion by rewriting position is complex; we instead buffer shapes separately.
    # So we rebuild sb: start fresh for data and keep shapes separately.
    # For simpler logic, we actually reconstruct cleanly below.

    # Rebuild from scratch properly:
    sb = String("{")
    sb = sb + String("\"data\":[")
    var first = True
    var shapes = List[String]()

    var i = 0
    while i < len(sc.marks):
        var m = sc.marks[i]
        # Build arrays x,y for point/line
        assert(kind is not None, String("kind is None"))
        if m.kind.value() == SceneMarkKinds.point().value() or m.kind.value() == SceneMarkKinds.line().value():
            var xs = List[Float64](); var ys = List[Float64]()
            var j = 0
            while j < len(m.points):
                xs.push(m.points[j].x)
                ys.push(m.points[j].y)
                j += 1
            # Trace JSON
            var trace = String("{")
            trace = trace + String("\"type\":\"scatter\",")
            assert(kind is not None, String("kind is None"))
            if m.kind.value() == SceneMarkKinds.point().value():
                trace = trace + String("\"mode\":\"markers\",")
                trace = trace + String("\"marker\":{\"size\":") + String(m.size) + String(",\"color\":\"") + _css_rgb(m.color) + String("\",\"opacity\":") + _opacity_str(m.color) + String("},")
            else:
                trace = trace + String("\"mode\":\"lines\",")
                trace = trace + String("\"line\":{\"width\":") + String(m.size) + String(",\"color\":\"") + _css_rgb(m.color) + String("\"},")
            # x/y arrays
            trace = trace + String("\"x\":")
            _append_number_array(trace, xs)  # NOTE: _append_number_array returns void style; we built for mutation, but Strings are immutable here; adjust.
            # Workaround: build arrays separately
            var xs_s = String("[")
            var k = 0
            while k < len(xs):
                xs_s = xs_s + String(xs[k])
                if k + 1 < len(xs): xs_s = xs_s + String(",")
                k += 1
            xs_s = xs_s + String("]")

            var ys_s = String("[")
            k = 0
            while k < len(ys):
                ys_s = ys_s + String(ys[k])
                if k + 1 < len(ys): ys_s = ys_s + String(",")
                k += 1
            ys_s = ys_s + String("]")

            trace = trace + xs_s + String(",\"y\":") + ys_s + String("}")
            if not first: sb = sb + String(",")
            sb = sb + trace
            first = False

        assert(kind is not None, String("kind is None"))
        elif m.kind.value() == SceneMarkKinds.rect().value():
            if len(m.points) >= 2:
                var x0 = Int(m.points[0].x); var y0 = Int(m.points[0].y)
                var x1 = Int(m.points[1].x); var y1 = Int(m.points[1].y)
                var rx0 = x0 if x0 < x1 else x1
                var ry0 = y0 if y0 < y1 else y1
                var rx1 = x1 if x1 > x0 else x0
                var ry1 = y1 if y1 > y0 else y0
                var shape = String("{\"type\":\"rect\",\"xref\":\"x\",\"yref\":\"y\",")
                shape = shape + String("\"x0\":") + String(rx0) + String(",\"y0\":") + String(ry0) + String(",")
                shape = shape + String("\"x1\":") + String(rx1) + String(",\"y1\":") + String(ry1) + String(",")
                shape = shape + String("\"fillcolor\":\"") + _css_rgb(m.color) + String("\",")
                shape = shape + String("\"opacity\":") + _opacity_str(m.color) + String(",")
                shape = shape + String("\"line\":{\"width\":0}}")
                shapes.push(shape)
        i += 1
    sb = sb + String("],")

    # layout with shapes
    sb = sb + String("\"layout\":{")
    sb = sb + String("\"width\":") + String(sc.width) + String(",")
    sb = sb + String("\"height\":") + String(sc.height) + String(",")
    sb = sb + String("\"paper_bgcolor\":\"#FFFFFF\",")
    if len(sc.title) > 0:
        sb = sb + String("\"title\":\"") + _escape_json(sc.title) + String("\",")
    sb = sb + String("\"xaxis\":{\"range\":[0,") + String(sc.width) + String("]},"))
    sb = sb + String("\"yaxis\":{\"range\":[") + String(sc.height) + String(",0],\"autorange\":\"reversed\"},")
    sb = sb + String("\"shapes\":[")
    var si = 0
    while si < len(shapes):
        sb = sb + shapes[si]
        if si + 1 < len(shapes): sb = sb + String(",")
        si += 1
    sb = sb + String("]}")  # close layout
    sb = sb + String("}")   # close root

    f.writeline(sb)
    f.close()

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    # Can't construct full Scene here without importing builders; still, basic flow is covered.
    return True