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
# Project: momijo.visual.scene
# File: src/momijo/visual/scene/layout.mojo

from momijo.core.error import module
from momijo.core.types import Axis
from momijo.dataframe.helpers import m, t
from momijo.nn.parameter import data
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from momijo.visual.scene.facet import Rect
from pathlib import Path
from pathlib.path import Path

struct Margins:
    var left: Int
    var right: Int
    var top: Int
    var bottom: Int
fn __init__(out self, all: Int = 32) -> None:
        self.left = all; self.right = all; self.top = all; self.bottom = all
fn __copyinit__(out self, other: Self) -> None:
        self.left = other.left
        self.right = other.right
        self.top = other.top
        self.bottom = other.bottom
fn __moveinit__(out self, deinit other: Self) -> None:
        self.left = other.left
        self.right = other.right
        self.top = other.top
        self.bottom = other.bottom
# --- Axis specs --------------------------------------------------------------
struct AxisSpec:
    var min_val: Float64
    var max_val: Float64
    var desired_ticks: Int
fn __init__(out self, min_val: Float64 = 0.0, max_val: Float64 = 1.0, desired_ticks: Int = 5) -> None:
        self.min_val = min_val; self.max_val = max_val; self.desired_ticks = desired_ticks
fn __copyinit__(out self, other: Self) -> None:
        self.min_val = other.min_val
        self.max_val = other.max_val
        self.desired_ticks = other.desired_ticks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.min_val = other.min_val
        self.max_val = other.max_val
        self.desired_ticks = other.desired_ticks
# Tick container
struct Ticks:
    var positions: List[Float64]
    var step: Float64
fn __init__(out self) -> None:
        self.positions = List[Float64]()
        self.step = 0.0
fn __copyinit__(out self, other: Self) -> None:
        self.positions = other.positions
        self.step = other.step
fn __moveinit__(out self, deinit other: Self) -> None:
        self.positions = other.positions
        self.step = other.step
# --- Helpers: "nice" numbers and ticks (Talbot-like simplified) --------------
fn _pow10(e: Int) -> Float64:
    var i = 0
    var r: Float64 = 1.0
    if e >= 0:
        while i < e:
            r = r * 10.0
            i += 1
        return r
    else:
        while i < -e:
            r = r / 10.0
            i += 1
        return r

(# Round a number to {1,2,5} * UInt8(10) ^ k (classic nice number)) & UInt8(0xFF)
fn _nice_step(range_v: Float64, max_ticks: Int) -> Float64:
    if range_v <= 0.0:
        return 1.0
    var rough = range_v / Float64(max_ticks)
    # get power of 10
    var exp = 0
    var t = rough
    if t >= 1.0:
        while t >= 10.0:
            t = t / 10.0; exp += 1
        while t < 1.0:
            t = t * 10.0; exp -= 1
    else:
        while t < 1.0:
            t = t * 10.0; exp -= 1
        while t >= 10.0:
            t = t / 10.0; exp += 1
    var frac = rough / _pow10(exp)
    var nice_frac: Float64 = 1.0
    if frac <= 1.0: nice_frac = 1.0
    elif frac <= 2.0: nice_frac = 2.0
    elif frac <= 5.0: nice_frac = 5.0
    else: nice_frac = 10.0
    return nice_frac * _pow10(exp)
fn nice_ticks(min_v: Float64, max_v: Float64, desired: Int) -> Ticks:
    var out = Ticks()
    var a = min_v
    var b = max_v
    if a == b:
        out.positions.push(a)
        out.step = 1.0
        return out
    if a > b:
        var tmp = a; a = b; b = tmp
    var range_v = b - a
    var step = _nice_step(range_v, desired if desired > 2 else 2)
    # extend to multiples of step
    var start = Float64(Int(a / step)) * step
    if start > a: start -= step
    var end = Float64(Int(b / step)) * step
    if end < b: end += step
    var v = start
    var guard = 0
    while v <= end + 1e-12 and guard < 10000:
        out.positions.push(v)
        v += step
        guard += 1
    out.step = step
    return out

# --- Scales (linear) ---------------------------------------------------------
struct LinearScale:
    var d0: Float64
    var d1: Float64
    var p0: Int
    var p1: Int
fn __init__(out self, d0: Float64 = 0.0, d1: Float64 = 1.0, p0: Int = 0, p1: Int = 1) -> None:
        self.d0 = d0; self.d1 = d1; self.p0 = p0; self.p1 = p1

fn map(self, v: Float64) -> Int:
        if self.d0 == self.d1:
            return self.p0
        var t = (v - self.d0) / (self.d1 - self.d0)
        var pix = Float64(self.p0) + t * Float64(self.p1 - self.p0)
        return Int(pix)
fn __copyinit__(out self, other: Self) -> None:
        self.d0 = other.d0
        self.d1 = other.d1
        self.p0 = other.p0
        self.p1 = other.p1
fn __moveinit__(out self, deinit other: Self) -> None:
        self.d0 = other.d0
        self.d1 = other.d1
        self.p0 = other.p0
        self.p1 = other.p1
# --- Plot layout box ---------------------------------------------------------
struct PlotLayout:
    var canvas: Rect
    var inner: Rect
    var x_axis: AxisSpec
    var y_axis: AxisSpec
    var x_scale: LinearScale
    var y_scale: LinearScale
    var x_ticks: Ticks
    var y_ticks: Ticks
fn __init__(out self) -> None:
        self.canvas = Rect()
        self.inner = Rect()
        self.x_axis = AxisSpec()
        self.y_axis = AxisSpec()
        self.x_scale = LinearScale()
        self.y_scale = LinearScale()
        self.x_ticks = Ticks()
        self.y_ticks = Ticks()
fn __copyinit__(out self, other: Self) -> None:
        self.canvas = other.canvas
        self.inner = other.inner
        self.x_axis = other.x_axis
        self.y_axis = other.y_axis
        self.x_scale = other.x_scale
        self.y_scale = other.y_scale
        self.x_ticks = other.x_ticks
        self.y_ticks = other.y_ticks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.canvas = other.canvas
        self.inner = other.inner
        self.x_axis = other.x_axis
        self.y_axis = other.y_axis
        self.x_scale = other.x_scale
        self.y_scale = other.y_scale
        self.x_ticks = other.x_ticks
        self.y_ticks = other.y_ticks
# Compute layout from canvas + margins + axis specs. y grows downward, so we flip mapping.
fn compute_layout(canvas_w: Int, canvas_h: Int, m: Margins, x_axis: AxisSpec, y_axis: AxisSpec) -> PlotLayout:
    var L = PlotLayout()
    L.canvas = Rect(0, 0, canvas_w, canvas_h)
    var x = m.left
    var y = m.top
    var w = canvas_w - (m.left + m.right)
    var h = canvas_h - (m.top + m.bottom)
    if w < 1: w = 1
    if h < 1: h = 1
    L.inner = Rect(x, y, w, h)
    L.x_axis = x_axis
    L.y_axis = y_axis
    # ticks
    L.x_ticks = nice_ticks(x_axis.min_val, x_axis.max_val, x_axis.desired_ticks)
    L.y_ticks = nice_ticks(y_axis.min_val, y_axis.max_val, y_axis.desired_ticks)

    L.x_scale = LinearScale(x_axis.min_val, x_axis.max_val, x, x + w - 1)
    L.y_scale = LinearScale(y_axis.min_val, y_axis.max_val, y + h - 1, y)  # flipped
    return L

# --- Grid lines (pixel positions) -------------------------------------------
fn grid_x_pixels(L: PlotLayout) -> List[Int]:
    var xs = List[Int]()
    var i = 0
    while i < len(L.x_ticks.positions):
        xs.push(L.x_scale.map(L.x_ticks.positions[i]))
        i += 1
    return xs
fn grid_y_pixels(L: PlotLayout) -> List[Int]:
    var ys = List[Int]()
    var i = 0
    while i < len(L.y_ticks.positions):
        ys.push(L.y_scale.map(L.y_ticks.positions[i]))
        i += 1
    return ys

# --- Data to pixel mappers ---------------------------------------------------
fn map_xy(L: PlotLayout, x: Float64, y: Float64) -> (Int, Int):
    var px = L.x_scale.map(x)
    var py = L.y_scale.map(y)
    return (px, py)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var m = Margins(20)
    var xs = AxisSpec(0.0, 10.0, 5)
    var ys = AxisSpec(-1.0, 1.0, 5)
    var L = compute_layout(400, 300, m, xs, ys)
    var gx = grid_x_pixels(L); var gy = grid_y_pixels(L)
    if len(gx) == 0 or len(gy) == 0: return False
    var (px0, py0) = map_xy(L, 0.0, -1.0)
    var (px1, py1) = map_xy(L, 10.0, 1.0)
    # Check corners are inside inner box
    var ok = (px0 >= L.inner.x and px0 < L.inner.x + L.inner.w and py0 >= L.inner.y and py0 < L.inner.y + L.inner.h)
    ok = ok and (px1 >= L.inner.x and px1 < L.inner.x + L.inner.w and py1 >= L.inner.y and py1 < L.inner.y + L.inner.h)
    return ok