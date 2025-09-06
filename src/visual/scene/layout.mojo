# ============================================================================
#  Momijo Visualization - scene/layout.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.scene.scene import ColorRGBA

struct AxisSpec:
    var show: Bool
    var ticks: Int
    var grid: Bool
    var label: String
    var log_scale: Bool
    var formatter: String   # e.g., ".2f", "sci"
    fn __init__(out self):
        self.show = True
        self.ticks = 5
        self.grid = True
        self.label = String("")
        self.log_scale = False
        self.formatter = String("")

struct AxisTick:
    var pos: Float64
    var label: String
    fn __init__(out self, pos: Float64, label: String):
        self.pos = pos
        self.label = label

struct AxisLayout:
    var ticks: List[AxisTick]
    fn __init__(out self):
        self.ticks = List[AxisTick]()

fn linspace(a: Float64, b: Float64, n: Int) -> List[Float64]:
    var out = List[Float64]()
    if n <= 1:
        out.append(a)
        return out
    var step = (b - a) / Float64(n - 1)
    var i = 0
    while i < n:
        out.append(a + Float64(i) * step)
        i += 1
    return out

fn format_float(v: Float64, spec: String) -> String:
    # very naive formatter: fixed with 2 decimals when ".2f"
    if spec == String(".2f"):
        return String(round(v * 100.0) / 100.0)
    return String(v)
