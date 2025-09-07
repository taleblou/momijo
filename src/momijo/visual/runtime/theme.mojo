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
# File: src/momijo/visual/runtime/theme.mojo

from momijo.dataframe.diagnostics import safe
from momijo.dataframe.helpers import between, t
from momijo.utils.result import g
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from momijo.visual.scene.scene import ColorRGBA
from pathlib import Path
from pathlib.path import Path
from sys import platform

fn _clamp8(x: Int) -> Int:
    var v = x
    if v < 0: v = 0
    if v > 255: v = 255
    return v
fn rgb(r: Int, g: Int, b: Int) -> ColorRGBA:
    return ColorRGBA(_clamp8(r), _clamp8(g), _clamp8(b), 255)
fn rgba(r: Int, g: Int, b: Int, a: Int) -> ColorRGBA:
    return ColorRGBA(_clamp8(r), _clamp8(g), _clamp8(b), _clamp8(a))

# Simple linear blend between two colors (0..1 mapped via t in [0, 1] as 0..1000 int steps)
fn lerp(c0: ColorRGBA, c1: ColorRGBA, t_milli: Int) -> ColorRGBA:
    # t_milli in [0,1000]
    var t = t_milli
    if t < 0: t = 0
    if t > 1000: t = 1000
    var r = (c0.r * (1000 - t) + c1.r * t) / 1000
    var g = (c0.g * (1000 - t) + c1.g * t) / 1000
    var b = (c0.b * (1000 - t) + c1.b * t) / 1000
    var a = (c0.a * (1000 - t) + c1.a * t) / 1000
    return ColorRGBA(r, g, b, a)

# --- Theme struct ------------------------------------------------------------
struct Theme:
    var name: String
    var background: ColorRGBA
    var axis: ColorRGBA
    var grid: ColorRGBA
    var text: ColorRGBA
    var palette_cat: List[ColorRGBA]   # categorical palette
    var gradient_lo: ColorRGBA         # for continuous scale
    var gradient_hi: ColorRGBA
fn __init__(out self) -> None:
        self.name = String("custom")
        self.background = rgb(255,255,255)
        self.axis = rgb(40, 40, 40)
        self.grid = rgba(120,120,120,60)
        self.text = rgb(20, 20, 20)
        self.palette_cat = List[ColorRGBA]()
        self.gradient_lo = rgb(68, 1, 84)   # approx viridis start
        self.gradient_hi = rgb(253, 231, 37) # approx viridis end
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.background = other.background
        self.axis = other.axis
        self.grid = other.grid
        self.text = other.text
        self.palette_cat = other.palette_cat
        self.gradient_lo = other.gradient_lo
        self.gradient_hi = other.gradient_hi
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.background = other.background
        self.axis = other.axis
        self.grid = other.grid
        self.text = other.text
        self.palette_cat = other.palette_cat
        self.gradient_lo = other.gradient_lo
        self.gradient_hi = other.gradient_hi
# Generate N colors by cycling the categorical palette (safe for small N).
fn theme_color_at(t: Theme, idx: Int) -> ColorRGBA:
    var n = len(t.palette_cat)
    if n == 0:
        return rgb(30, 136, 229)  # fallback blue
    var k = idx % n
    if k < 0: k += n
    return t.palette_cat[k]

# Sample the continuous gradient with t in [0,1000]
fn theme_grad_sample(t: Theme, t_milli: Int) -> ColorRGBA:
    return lerp(t.gradient_lo, t.gradient_hi, t_milli)

# --- Preset palettes ---------------------------------------------------------
fn _palette_scientific() -> List[ColorRGBA]:
    var p = List[ColorRGBA]()
    # Distinct, colorblind-friendly-ish set
    p.push(rgb(68, 119, 170))   # blue
    p.push(rgb(221, 204, 119))  # sand
    p.push(rgb(204, 102, 119))  # rose
    p.push(rgb(102, 204, 238))  # light blue
    p.push(rgb(34, 136, 51))    # green
    p.push(rgb(170, 51, 119))   # magenta
    p.push(rgb(187, 187, 187))  # gray
    return p
fn _palette_dark() -> List[ColorRGBA]:
    var p = List[ColorRGBA]()
    p.push(rgb(78, 201, 176))
    p.push(rgb(97, 175, 239))
    p.push(rgb(198, 120, 221))
    p.push(rgb(209, 154, 102))
    p.push(rgb(152, 195, 121))
    p.push(rgb(224, 108, 117))
    p.push(rgb(171, 178, 191))
    return p
fn _palette_publisher() -> List[ColorRGBA]:
    var p = List[ColorRGBA]()
    p.push(rgb(33, 150, 243))   # primary
    p.push(rgb(233, 30, 99))    # accent
    p.push(rgb(76, 175, 80))    # success
    p.push(rgb(255, 193, 7))    # warning
    p.push(rgb(244, 67, 54))    # danger
    p.push(rgb(121, 85, 72))    # brown
    p.push(rgb(96, 125, 139))   # blue-gray
    return p

# --- Preset themes -----------------------------------------------------------
fn theme_scientific() -> Theme:
    var t = Theme()
    t.name = String("scientific")
    t.background = rgb(255,255,255)
    t.axis = rgb(20, 20, 20)
    t.grid = rgba(120,120,120,60)
    t.text = rgb(10, 10, 10)
    t.palette_cat = _palette_scientific()
    # Viridis-ish
    t.gradient_lo = rgb(68, 1, 84)
    t.gradient_hi = rgb(253, 231, 37)
    return t
fn theme_dark() -> Theme:
    var t = Theme()
    t.name = String("dark")
    t.background = rgb(18, 18, 18)
    t.axis = rgb(210, 210, 210)
    t.grid = rgba(160, 160, 160, 50)
    t.text = rgb(230, 230, 230)
    t.palette_cat = _palette_dark()

    t.gradient_lo = rgb(0, 122, 204)
    t.gradient_hi = rgb(78, 201, 176)
    return t
fn theme_publisher() -> Theme:
    var t = Theme()
    t.name = String("publisher")
    t.background = rgb(255, 255, 255)
    t.axis = rgb(60, 60, 60)
    t.grid = rgba(140, 140, 140, 60)
    t.text = rgb(30, 30, 30)
    t.palette_cat = _palette_publisher()

    t.gradient_lo = rgb(103, 58, 183)
    t.gradient_hi = rgb(255, 193, 7)
    return t

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var t = theme_scientific()
    # Check palette non-empty and gradient range
    var ok = len(t.palette_cat) > 0
    var c0 = theme_grad_sample(t, 0)
    var c1 = theme_grad_sample(t, 1000)
    ok = ok and (c0.r != c1.r or c0.g != c1.g or c0.b != c1.b)
    return ok