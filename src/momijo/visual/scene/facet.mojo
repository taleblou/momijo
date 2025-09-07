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
# File: src/momijo/visual/scene/facet.mojo

struct Rect:
    var x: Int
    var y: Int
    var w: Int
    var h: Int
fn __init__(out self, x: Int = 0, y: Int = 0, w: Int = 0, h: Int = 0) -> None:
        self.x = x; self.y = y; self.w = w; self.h = h
fn __copyinit__(out self, other: Self) -> None:
        self.x = other.x
        self.y = other.y
        self.w = other.w
        self.h = other.h
fn __moveinit__(out self, deinit other: Self) -> None:
        self.x = other.x
        self.y = other.y
        self.w = other.w
        self.h = other.h
# --- Facet specification -----------------------------------------------------
struct FacetSpec:
    var rows: Int
    var cols: Int
    var gutter_x: Int    # space between columns (pixels)
    var gutter_y: Int    # space between rows    (pixels)
    var margin_l: Int    # outer margins (pixels)
    var margin_r: Int
    var margin_t: Int
    var margin_b: Int
fn __init__(out self,
                rows: Int = 1, cols: Int = 1,
                gutter_x: Int = 8, gutter_y: Int = 8,
                margin: Int = 16) -> None:
        self.rows = rows; self.cols = cols
        self.gutter_x = gutter_x; self.gutter_y = gutter_y
        self.margin_l = margin; self.margin_r = margin
        self.margin_t = margin; self.margin_b = margin
fn __copyinit__(out self, other: Self) -> None:
        self.rows = other.rows
        self.cols = other.cols
        self.gutter_x = other.gutter_x
        self.gutter_y = other.gutter_y
        self.margin_l = other.margin_l
        self.margin_r = other.margin_r
        self.margin_t = other.margin_t
        self.margin_b = other.margin_b
fn __moveinit__(out self, deinit other: Self) -> None:
        self.rows = other.rows
        self.cols = other.cols
        self.gutter_x = other.gutter_x
        self.gutter_y = other.gutter_y
        self.margin_l = other.margin_l
        self.margin_r = other.margin_r
        self.margin_t = other.margin_t
        self.margin_b = other.margin_b
# --- Grid computation --------------------------------------------------------
# Returns a row-major list of cell rects of length rows*cols.
fn facet_grid(canvas_w: Int, canvas_h: Int, spec: FacetSpec) -> List[Rect]:
    var out = List[Rect]()
    if canvas_w <= 0 or canvas_h <= 0 or spec.rows <= 0 or spec.cols <= 0:
        return out

    var W = canvas_w - (spec.margin_l + spec.margin_r)
    var H = canvas_h - (spec.margin_t + spec.margin_b)
    if W <= 0 or H <= 0:
        return out

    var total_gx = spec.gutter_x * (spec.cols - 1)
    var total_gy = spec.gutter_y * (spec.rows - 1)
    var cw = (W - total_gx) / spec.cols
    var ch = (H - total_gy) / spec.rows
    if cw <= 0 or ch <= 0:
        return out

    var r = 0
    while r < spec.rows:
        var c = 0
        while c < spec.cols:
            var x = spec.margin_l + c * (cw + spec.gutter_x)
            var y = spec.margin_t + r * (ch + spec.gutter_y)
            out.push(Rect(x, y, cw, ch))
            c += 1
        r += 1
    return out

# Indexing helper (row-major)
fn facet_index(rows: Int, cols: Int, r: Int, c: Int) -> Int:
    if r < 0 or r >= rows or c < 0 or c >= cols: return -1
    return r * cols + c

# --- Normalized coordinate mapper -------------------------------------------
# Maps (nx, ny) in [0,1]x[0,1] to pixel coordinates inside a rect with padding.
struct FacetMapper:
    var pad_left: Int
    var pad_right: Int
    var pad_top: Int
    var pad_bottom: Int
fn __init__(out self, pad: Int = 24) -> None:
        self.pad_left = pad; self.pad_right = pad
        self.pad_top = pad;  self.pad_bottom = pad
fn inner(self, cell: Rect) -> Rect:
        var x = cell.x + self.pad_left
        var y = cell.y + self.pad_top
        var w = cell.w - (self.pad_left + self.pad_right)
        var h = cell.h - (self.pad_top + self.pad_bottom)
        if w < 1: w = 1
        if h < 1: h = 1
        return Rect(x, y, w, h)

fn map(self, cell: Rect, nx: Float64, ny: Float64) -> (Int, Int):
        var inr = self.inner(cell)
        var fx = nx
        var fy = ny
        if fx < 0.0: fx = 0.0
        if fx > 1.0: fx = 1.0
        if fy < 0.0: fy = 0.0
        if fy > 1.0: fy = 1.0
        var x = inr.x + Int(fx * Float64(inr.w - 1))
        var y = inr.y + Int((1.0 - fy) * Float64(inr.h - 1))  # flip to typical plot coord
        return (x, y)
fn __copyinit__(out self, other: Self) -> None:
        self.pad_left = other.pad_left
        self.pad_right = other.pad_right
        self.pad_top = other.pad_top
        self.pad_bottom = other.pad_bottom
fn __moveinit__(out self, deinit other: Self) -> None:
        self.pad_left = other.pad_left
        self.pad_right = other.pad_right
        self.pad_top = other.pad_top
        self.pad_bottom = other.pad_bottom
# --- Convenience: grid + mapper ---------------------------------------------
struct FacetLayout:
    var spec: FacetSpec
    var cells: List[Rect]
    var mapper: FacetMapper
fn __init__(out self, spec: FacetSpec, canvas_w: Int, canvas_h: Int, pad: Int = 24) -> None:
        self.spec = spec
        self.cells = facet_grid(canvas_w, canvas_h, spec)
        self.mapper = FacetMapper(pad)
fn __copyinit__(out self, other: Self) -> None:
        self.spec = other.spec
        self.cells = other.cells
        self.mapper = other.mapper
fn __moveinit__(out self, deinit other: Self) -> None:
        self.spec = other.spec
        self.cells = other.cells
        self.mapper = other.mapper
# --- Example: place series points into a given cell --------------------------
# Given a list of normalized points, project them to pixel coordinates in cell idx.
struct NormPoint:
    var x: Float64
    var y: Float64
fn __init__(out self, x: Float64, y: Float64) -> None:
        self.x = x; self.y = y
fn __copyinit__(out self, other: Self) -> None:
        self.x = other.x
        self.y = other.y
fn __moveinit__(out self, deinit other: Self) -> None:
        self.x = other.x
        self.y = other.y
fn facet_points(layout: FacetLayout, idx: Int, pts: List[NormPoint]) -> (List[Int], List[Int]):
    var xs = List[Int]()
    var ys = List[Int]()
    if idx < 0 or idx >= len(layout.cells): return (xs, ys)
    var cell = layout.cells[idx]
    var i = 0
    while i < len(pts):
        var (px, py) = layout.mapper.map(cell, pts[i].x, pts[i].y)
        xs.push(px); ys.push(py)
        i += 1
    return (xs, ys)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var spec = FacetSpec(2, 2, 10, 10, 20)
    var grid = facet_grid(400, 300, spec)
    if len(grid) != 4: return False
    var fl = FacetLayout(spec, 400, 300, 16)
    if len(fl.cells) != 4: return False
    var pts = List[NormPoint]()
    pts.push(NormPoint(0.0, 0.0)); pts.push(NormPoint(1.0, 1.0))
    var (xs, ys) = facet_points(fl, 0, pts)
    return (len(xs) == 2 and len(ys) == 2)