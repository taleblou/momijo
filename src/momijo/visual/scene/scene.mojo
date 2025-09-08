# Project:      Momijo
# Module:       src.momijo.visual.scene.scene
# File:         scene.mojo
# Path:         src/momijo/visual/scene/scene.mojo
#
# Description:  src.momijo.visual.scene.scene — focused Momijo functionality with a stable public API.
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
#   - Structs: ColorRGBA, Point2D, SceneMarkKind, SceneMarkKinds, Mark, Scene
#   - Key functions: __init__, __copyinit__, __moveinit__, __init__, __copyinit__, __moveinit__, __init__, __copyinit__ ...
#   - Static methods present.


struct ColorRGBA:
    var r: Int
    var g: Int
    var b: Int
    var a: Int
fn __init__(out self, r: Int = 0, g: Int = 0, b: Int = 0, a: Int = 255) -> None:
        self.r = r; self.g = g; self.b = b; self.a = a
fn __copyinit__(out self, other: Self) -> None:
        self.r = other.r
        self.g = other.g
        self.b = other.b
        self.a = other.a
fn __moveinit__(out self, deinit other: Self) -> None:
        self.r = other.r
        self.g = other.g
        self.b = other.b
        self.a = other.a
# --- Geometry ----------------------------------------------------------------
struct Point2D:
    var x: Float64
    var y: Float64
fn __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0) -> None:
        self.x = x; self.y = y
fn __copyinit__(out self, other: Self) -> None:
        self.x = other.x
        self.y = other.y
fn __moveinit__(out self, deinit other: Self) -> None:
        self.x = other.x
        self.y = other.y
# --- Mark kinds --------------------------------------------------------------
struct SceneMarkKind:
    var value: Int
assert(self is not None, String("self is None"))
fn __init__(out self, value: Int) -> None: self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct SceneMarkKinds:
    @staticmethod
fn point() -> SceneMarkKind: return SceneMarkKind(0)
    @staticmethod
fn line() -> SceneMarkKind:  return SceneMarkKind(1)
    @staticmethod
fn rect() -> SceneMarkKind:  return SceneMarkKind(2)
    @staticmethod
fn text() -> SceneMarkKind:  return SceneMarkKind(3)
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# --- Mark --------------------------------------------------------------------
struct Mark:
    var kind: SceneMarkKind
    var color: ColorRGBA
    var size: Int                  # radius for point; stroke width for line; font size for text
    var points: List[Point2D]      # geometry payload (polyline; rect uses 2 points; point uses many)
    var label: String              # for text marks
fn __init__(out self) -> None:
        self.kind = SceneMarkKinds.point()
        self.color = ColorRGBA(30, 136, 229, 255)  # default blue
        self.size = 2
        self.points = List[Point2D]()
        self.label = String("")
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.color = other.color
        self.size = other.size
        self.points = other.points
        self.label = other.label
fn __moveinit__(out self, deinit other: Self) -> None:
        self.kind = other.kind
        self.color = other.color
        self.size = other.size
        self.points = other.points
        self.label = other.label
# --- Scene -------------------------------------------------------------------
struct Scene:
    var width: Int
    var height: Int
    var background: ColorRGBA
    var marks: List[Mark]
    var title: String
    var x_title: String
    var y_title: String
fn __init__(out self, width: Int = 640, height: Int = 480) -> None:
        self.width = width
        self.height = height
        self.background = ColorRGBA(255,255,255,255)
        self.marks = List[Mark]()
        self.title = String("")
        self.x_title = String("")
        self.y_title = String("")

    # --- Add helpers ---------------------------------------------------------
fn add_mark(mut self, m: Mark) -> None:
        self.marks.push(m)

    # Add many points (scatter). size = radius px.
fn add_points(mut self, pts: List[Point2D], color: ColorRGBA, size: Int = 2) -> None:
        var m = Mark()
        m.kind = SceneMarkKinds.point()
        m.color = color
        m.size = size
        m.points = pts
        self.marks.push(m)

    # Add a single point.
fn add_point(mut self, x: Float64, y: Float64, color: ColorRGBA, size: Int = 2) -> None:
        var pts = List[Point2D]()
        pts.push(Point2D(x, y))
        self.add_points(pts, color, size)

    # Add a polyline (connected). size = stroke width px.
fn add_polyline(mut self, pts: List[Point2D], color: ColorRGBA, stroke_px: Int = 1) -> None:
        if len(pts) == 0: return
        var m = Mark()
        m.kind = SceneMarkKinds.line()
        m.color = color
        m.size = stroke_px
        m.points = pts
        self.marks.push(m)

    # Add rect by corners (x0,y0)-(x1,y1). Color used as fill; size optional as border width (currently unused).
fn add_rect(mut self, x0: Float64, y0: Float64, x1: Float64, y1: Float64, color: ColorRGBA, border_px: Int = 0) -> None:
        var m = Mark()
        m.kind = SceneMarkKinds.rect()
        m.color = color
        m.size = border_px
        m.points = List[Point2D]()
        m.points.push(Point2D(x0, y0))
        m.points.push(Point2D(x1, y1))
        self.marks.push(m)

    # Add text at (x,y) with font size in px. (Rendering depends on backend support.)
fn add_text(mut self, x: Float64, y: Float64, label: String, color: ColorRGBA, font_px: Int = 12) -> None:
        var m = Mark()
        m.kind = SceneMarkKinds.text()
        m.color = color
        m.size = font_px
        m.label = label
        m.points = List[Point2D]()
        m.points.push(Point2D(x, y))
        self.marks.push(m)
fn __copyinit__(out self, other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.background = other.background
        self.marks = other.marks
        self.title = other.title
        self.x_title = other.x_title
        self.y_title = other.y_title
fn __moveinit__(out self, deinit other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.background = other.background
        self.marks = other.marks
        self.title = other.title
        self.x_title = other.x_title
        self.y_title = other.y_title
# --- Convenience: color helpers ---------------------------------------------
fn rgb(r: Int, g: Int, b: Int) -> ColorRGBA:
    return ColorRGBA(r, g, b, 255)
fn rgba(r: Int, g: Int, b: Int, a: Int) -> ColorRGBA:
    return ColorRGBA(r, g, b, a)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var sc = Scene(320, 240)
    sc.title = String("demo")
    sc.add_point(10.0, 10.0, rgb(200,0,0), 3)
    var pts = List[Point2D]()
    pts.push(Point2D(20.0, 20.0)); pts.push(Point2D(100.0, 50.0)); pts.push(Point2D(200.0, 120.0))
    sc.add_polyline(pts, rgb(0,0,0), 1)
    sc.add_rect(30.0, 30.0, 80.0, 70.0, rgba(0,128,255,255), 0)
    sc.add_text(5.0, 15.0, String("hello"), rgb(0,0,0), 12)
    # basic sanity
    return (len(sc.marks) == 4) and (sc.width == 320) and (sc.height == 240)