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
# Project: momijo.visual.spec.exporters
# File: src/momijo/visual/spec/exporters/vega_lite_json.mojo

from io.file import open
from momijo.core.device import kind
from momijo.core.error import module
from momijo.core.traits import one
from momijo.dataframe.helpers import close, m, t
from momijo.dataframe.logical_plan import sort
from momijo.nn.parameter import data
from momijo.utils.result import f
from momijo.visual.scene.scene import Scene, SceneMarkKinds
from pathlib import Path
from pathlib.path import Path

# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.spec.exporters.vega_lite_json
# File:         vega_lite_json.mojo
# Path:         momijo/visual/spec/exporters/vega_lite_json.mojo
#
# Description:  Core module 'vega lite jso' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================
fn _escape_json(s: String) -> String:
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
# NOTE: Removed duplicate definition of `_css_rgb`; use `from momijo.visual.spec.exporters.plotly_json import _css_rgb`
fn _opacity_str(c: ColorRGBA) -> String:
    if c.a >= 255: return String("1")
    if c.a <= 0: return String("0")
    var milli = (c.a * 1000) / 255
    if milli >= 1000: milli = 999
    return String("0.") + (String("000") + String(milli))[-3:len(String("000") + String(milli))]

# Notes:
#  - Coordinates are interpreted as pixel space; we fix the view size to scene width/height,
#    and declare linear axes with fixed domain [0,width] and [height,0] (y reversed via sort/order).

#    Rects are drawn as an extra layer using "rect" mark with x/x2/y/y2 fields.
fn write_vega_lite_json(sc: Scene, path: String) -> None:
    var f = open(path, String("w"))
    if f.is_null(): return

    # Begin spec
    var sb = String("{")
    sb = sb + String("\"$schema\":\"https://vega.github.io/schema/vega-lite/v5.json\",")
    if len(sc.title) > 0:
        sb = sb + String("\"title\":\"") + _escape_json(sc.title) + String("\",")
    sb = sb + String("\"width\":") + String(sc.width) + String(",")
    sb = sb + String("\"height\":") + String(sc.height) + String(",")
    sb = sb + String("\"layer\":[")

    var first_layer = True

    var i = 0
    while i < len(sc.marks):
        var m = sc.marks[i]
        # Build inline data rows depending on mark
        assert(kind is not None, String("kind is None"))
        if m.kind.value() == SceneMarkKinds.point().value() or m.kind.value() == SceneMarkKinds.line().value():
            var rows = String("[")
            var j = 0
            while j < len(m.points):
                if j > 0: rows = rows + String(",")
                rows = rows + String("{\"x\":") + String(m.points[j].x) + String(",\"y\":") + String(m.points[j].y) + String("}")
                j += 1
            rows = rows + String("]")

            var layer = String("{")
            layer = layer + String("\"data\":{\"values\":") + rows + String("},")
            layer = layer + String("\"encoding\":{")
            layer = layer + String("\"x\":{\"field\":\"x\",\"type\":\"quantitative\",\"scale\":{\"domain\":[0,") + String(sc.width) + String("]}},")
            layer = layer + String("\"y\":{\"field\":\"y\",\"type\":\"quantitative\",\"scale\":{\"domain\":[") + String(sc.height) + String(",0]}}},")
            # style
            assert(kind is not None, String("kind is None"))
            if m.kind.value() == SceneMarkKinds.point().value():
                layer = layer + String("\"mark\":{\"type\":\"point\",\"filled\":true,\"size\":") + String(m.size * m.size * 4) + String(",")
                layer = layer + String("\"color\":\"") + _css_rgb(m.color) + String("\",\"opacity\":") + _opacity_str(m.color) + String("}}")
            else:
                layer = layer + String("\"mark\":{\"type\":\"line\",\"stroke\":\"") + _css_rgb(m.color) + String("\",\"strokeWidth\":") + String(m.size) + String("}}")
            if not first_layer: sb = sb + String(",")
            sb = sb + layer
            first_layer = False

        assert(kind is not None, String("kind is None"))
        elif m.kind.value() == SceneMarkKinds.rect().value() and len(m.points) >= 2:
            var x0 = Int(m.points[0].x); var y0 = Int(m.points[0].y)
            var x1 = Int(m.points[1].x); var y1 = Int(m.points[1].y)
            var rx0 = x0 if x0 < x1 else x1
            var ry0 = y0 if y0 < y1 else y1
            var rx1 = x1 if x1 > x0 else x0
            var ry1 = y1 if y1 > y0 else y0
            var rows2 = String("[{\"x\":") + String(rx0) + String(",\"x2\":") + String(rx1) + String(",\"y\":") + String(ry0) + String(",\"y2\":") + String(ry1) + String("}]")
            var layer2 = String("{")
            layer2 = layer2 + String("\"data\":{\"values\":") + rows2 + String("},")
            layer2 = layer2 + String("\"encoding\":{")
            layer2 = layer2 + String("\"x\":{\"field\":\"x\",\"type\":\"quantitative\",\"scale\":{\"domain\":[0,") + String(sc.width) + String("]}},")
            layer2 = layer2 + String("\"x2\":{\"field\":\"x2\"},")
            layer2 = layer2 + String("\"y\":{\"field\":\"y\",\"type\":\"quantitative\",\"scale\":{\"domain\":[") + String(sc.height) + String(",0]}}," )
            layer2 = layer2 + String("\"y2\":{\"field\":\"y2\"}}," )
            layer2 = layer2 + String("\"mark\":{\"type\":\"rect\",\"opacity\":") + _opacity_str(m.color) + String(",\"fill\":\"") + _css_rgb(m.color) + String("\"}}")
            if not first_layer: sb = sb + String(",")
            sb = sb + layer2
            first_layer = False
        i += 1

    sb = sb + String("]}")  # close layer + root

    f.writeline(sb)
    f.close()

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    # Structure-only sanity
    return True