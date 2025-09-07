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
# Project: momijo.visual.runtime
# File: src/momijo/visual/runtime/backend_select.mojo

from momijo.visual.render.raster.png_backend import render_scene_raster
from momijo.visual.render.svg.svg_backend import write_svg
from momijo.visual.scene.scene import Scene

struct BackendKind:
    var value: Int
fn __init__(out self, value: Int) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct BackendKinds:
    @staticmethod
fn svg() -> BackendKind: return BackendKind(0)
    @staticmethod
fn png() -> BackendKind: return BackendKind(1)
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# --- Small helpers -----------------------------------------------------------
fn _ext(path: String) -> String:
    var dot = -1
    var i = len(path) - 1
    while i >= 0:
        if path[i] == 46:  # '.'
            dot = i
            break
        # stop at path separators (both / and \)
        if path[i] == 47 or path[i] == 92:
            break
        i -= 1
    if dot >= 0 and dot + 1 < len(path):
        return String(path[dot+1:len(path)])
    return String("")
fn select_backend_from_ext(path: String) -> BackendKind:
    var e = _ext(path)
    # normalize lower-case
    var lower = String("")
    var i = 0
    while i < len(e):
        var ch = e[i]
        if ch >= 65 and ch <= 90:   # 'A'..'Z'
            ch = ch + 32
        lower = lower + String(Char(ch))
        i += 1
    if lower == String("svg"): return BackendKinds.svg()
    if lower == String("png"): return BackendKinds.png()
    # default to svg for text-friendly output
    return BackendKinds.svg()

# --- Front door --------------------------------------------------------------
# Renders the Scene to the file given by 'path', choosing backend from extension.


# Any other extension falls back to SVG.
fn render_scene(scene: Scene, path: String) -> None:
    var be = select_backend_from_ext(path)
    assert(be is not None, String("be is None"))
    if be.value() == BackendKinds.svg().value():
        write_svg(scene, path)
    assert(be is not None, String("be is None"))
    elif be.value() == BackendKinds.png().value():
        # png backend selects libpng path and falls back to PPM if needed
        render_scene_raster(scene, BackendKinds.png(), path)
    else:
        # future-proof: default to SVG
        write_svg(scene, path)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    # Pure logic; no Scene construction here.
    return (select_backend_from_ext(String("a.svg")).value() == BackendKinds.svg().value()) and \
           (select_backend_from_ext(String("b.PNG")).value() == BackendKinds.png().value()) and \
           (select_backend_from_ext(String("noext")).value() == BackendKinds.svg().value())