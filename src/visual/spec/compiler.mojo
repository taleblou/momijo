# ============================================================================
#  Momijo Visualization - spec/compiler.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.spec.spec import Spec, MarkKinds
from momijo.visual.scene.scene import Scene, Mark, SceneMarkKinds, Point2D, ColorRGBA, LegendItem
from momijo.visual.scene.layout import AxisTick, linspace, format_float

fn _min_max(xs: List[Float64]) -> (Float64, Float64):
    var n = len(xs)
    if n == 0: return (0.0, 1.0)
    var mn = xs[0]
    var mx = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v < mn: mn = v
        if v > mx: mx = v
        i += 1
    if mn == mx:
        mx = mn + 1.0
    return (mn, mx)

fn _palette(idx: Int) -> ColorRGBA:
    var colors = [
        ColorRGBA(31, 119, 180, 1.0),
        ColorRGBA(255, 127, 14, 1.0),
        ColorRGBA(44, 160, 44, 1.0),
        ColorRGBA(214, 39, 40, 1.0),
        ColorRGBA(148, 103, 189, 1.0),
    ]
    var n = len(colors)
    var j = idx % n
    return colors[j]

fn _lerp(a: Float64, b: Float64, t: Float64) -> Float64:
    return a + (b - a) * t

fn _color_map_quant(v: Float64, vmin: Float64, vmax: Float64) -> ColorRGBA:
    var t = (v - vmin) / (vmax - vmin)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    var r = Int(_lerp(68.0, 220.0, t))
    var g = Int(_lerp(1.0, 220.0, t))
    var b = Int(_lerp(84.0, 47.0, t))
    return ColorRGBA(r,g,Int(b),1.0)

fn compile_spec_to_scene(spec: Spec) -> Scene:
    var scene = Scene(spec.width, spec.height, spec.padding)

    var xname = spec.enc.x.field
    var yname = spec.enc.y.field
    var cname = spec.enc.color.field

    var xs = spec.data.cols_num[xname]
    var ys = spec.data.cols_num[yname]
    var n = len(xs)
    if len(ys) < n: n = len(ys)

    var (xmin, xmax) = _min_max(xs)
    var (ymin, ymax) = _min_max(ys)

    var pad = Float64(spec.padding)
    var W = Float64(spec.width)  - 2.0 * pad
    var H = Float64(spec.height) - 2.0 * pad

    var use_log_x = spec.enc.x.dtype == String("log")
    var use_log_y = spec.enc.y.dtype == String("log")

    fn tr_x(x: Float64) -> Float64:
        if use_log_x and x > 0.0: return log(x)
        return x
    fn tr_y(y: Float64) -> Float64:
        if use_log_y and y > 0.0: return log(y)
        return y

    var (txmin, txmax) = _min_max([tr_x(xmin), tr_x(xmax)])
    var (tymin, tymax) = _min_max([tr_y(ymin), tr_y(ymax)])

    fn sx(x: Float64) -> Float64:
        return pad + (tr_x(x) - txmin) / (txmax - txmin) * W
    fn sy(y: Float64) -> Float64:
        return pad + H - (tr_y(y) - tymin) / (tymax - tymin) * H

    # titles
    scene.x_title = xname
    scene.y_title = yname

    # build base mark
    var mark = Mark(SceneMarkKinds.point())  # base layer
    if spec.mark.value == MarkKinds.line().value:
        mark = Mark(SceneMarkKinds.line())
        mark.size = 2.0
    elif spec.mark.value == MarkKinds.rect().value:
        mark = Mark(SceneMarkKinds.rect())
        mark.size = 1.0
    else:
        mark.size = 3.0

    var color_by_cat = Dict[String, ColorRGBA]()
    var color_idx = 0
    var c_quant = False
    var cmin = 0.0
    var cmax = 1.0

    if len(cname) > 0 and len(spec.data.cols_num[cname]) == n:
        # quantitative color
        c_quant = True
        var (cmin0,cmax0) = _min_max(spec.data.cols_num[cname])
        cmin = cmin0; cmax = cmax0
        scene.legend_cont.show = True
        scene.legend_cont.min_val = cmin
        scene.legend_cont.max_val = cmax

    var i = 0
    while i < n:
        var px = xs[i]; var py = ys[i]
        var sp = Point2D(sx(px), sy(py))
        mark.points.append(sp)

        if len(cname) > 0:
            if c_quant:
                var cv = spec.data.cols_num[cname][i]
                mark.color = _color_map_quant(cv, cmin, cmax)
            else:
                var cat = spec.data.cols_str[cname][i]
                if len(cat) > 0:
                    if not color_by_cat.contains(cat):
                        let c = _palette(color_idx)
                        color_by_cat[cat] = c
                        scene.legend.append(LegendItem(cat, c))
                        color_idx += 1
                    mark.color = color_by_cat[cat]
        i += 1

    scene.marks.append(mark)

    # Additional layers (simple: same scales)
    var li = 0
    while li < len(spec.layers):
        var L = spec.layers[li]
        var m2 = Mark(SceneMarkKinds.point())
        if L.mark.value == MarkKinds.line().value:
            m2 = Mark(SceneMarkKinds.line()); m2.size = 2.0
        elif L.mark.value == MarkKinds.rect().value:
            m2 = Mark(SceneMarkKinds.rect()); m2.size = 1.0
        else:
            m2.size = 3.0

        var xs2 = spec.data.cols_num[L.enc.x.field]
        var ys2 = spec.data.cols_num[L.enc.y.field]
        var n2 = len(xs2)
        if len(ys2) < n2: n2 = len(ys2)

        var i2 = 0
        while i2 < n2:
            var p2 = Point2D(sx(xs2[i2]), sy(ys2[i2]))
            m2.points.append(p2)
            i2 += 1
        scene.marks.append(m2)
        li += 1

    # ticks/grid
    var nt = 6
    var xt = linspace(txmin, txmax, nt)
    var yt = linspace(tymin, tymax, nt)
    var j = 0
    while j < len(xt):
        let xv = xt[j]
        var label = format_float(xv, String(".2f"))
        scene.x_axis.ticks.append(AxisTick(pad + (xv - txmin)/(txmax-txmin)*W, label))
        j += 1
    j = 0
    while j < len(yt):
        let yv = yt[j]
        var labely = format_float(yv, String(".2f"))
        scene.y_axis.ticks.append(AxisTick(pad + H - (yv - tymin)/(tymax-tymin)*H, labely))
        j += 1

    return scene
