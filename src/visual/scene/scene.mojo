# ============================================================================
#  Momijo Visualization - scene/scene.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.scene.layout import AxisSpec, AxisLayout

struct Point2D:
    var x: Float64
    var y: Float64
    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

struct ColorRGBA:
    var r: Int
    var g: Int
    var b: Int
    var a: Float64
    fn __init__(out self, r: Int, g: Int, b: Int, a: Float64 = 1.0):
        self.r = r; self.g = g; self.b = b; self.a = a

struct SceneMarkKind:
    var value: Int
    fn __init__(out self, value: Int): self.value = value
struct SceneMarkKinds:
    @staticmethod
    fn point() -> SceneMarkKind: return SceneMarkKind(0)
    @staticmethod
    fn line() -> SceneMarkKind: return SceneMarkKind(1)
    @staticmethod
    fn rect() -> SceneMarkKind: return SceneMarkKind(2)

struct Mark:
    var kind: SceneMarkKind
    var points: List[Point2D]
    var color: ColorRGBA
    var size: Float64   # radius for point; stroke width for line
    fn __init__(out self, kind: SceneMarkKind):
        self.kind = kind
        self.points = List[Point2D]()
        self.color = ColorRGBA(0,0,0,1.0)
        self.size = 2.0

struct LegendItem:
    var label: String
    var color: ColorRGBA
    fn __init__(out self, label: String, color: ColorRGBA):
        self.label = label
        self.color = color

struct ContinuousLegend:
    var show: Bool
    var min_val: Float64
    var max_val: Float64
    fn __init__(out self):
        self.show = False
        self.min_val = 0.0
        self.max_val = 1.0

struct Scene:
    var width: Int
    var height: Int
    var padding: Int
    var marks: List[Mark]
    var x_axis: AxisLayout
    var y_axis: AxisLayout
    var x_spec: AxisSpec
    var y_spec: AxisSpec
    var legend: List[LegendItem]          # categorical
    var legend_cont: ContinuousLegend     # quantitative color
    var x_title: String
    var y_title: String

    fn __init__(out self, width: Int, height: Int, padding: Int):
        self.width = width
        self.height = height
        self.padding = padding
        self.marks = List[Mark]()
        self.x_axis = AxisLayout()
        self.y_axis = AxisLayout()
        self.x_spec = AxisSpec()
        self.y_spec = AxisSpec()
        self.legend = List[LegendItem]()
        self.legend_cont = ContinuousLegend()
        self.x_title = String("")
        self.y_title = String("")
