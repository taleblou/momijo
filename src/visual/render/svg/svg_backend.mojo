# ============================================================================
#  Momijo Visualization - render/svg/svg_backend.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.scene.scene import Scene, Mark, SceneMarkKinds, Point2D, ColorRGBA, LegendItem
from momijo.visual.runtime.backend_select import BackendKind, BackendKinds
from momijo.visual.runtime.theme import theme_scientific, theme_dark, theme_publisher

fn _color_hex(c: ColorRGBA) -> String:
    var r = c.r; var g = c.g; var b = c.b
    fn hx(u: Int) -> String:
        var digits = String("0123456789ABCDEF")
        var hi = (u / 16) % 16
        var lo = u % 16
        var s = String("") + digits[hi:hi+1] + digits[lo:lo+1]
        return s
    return String("#") + hx(r) + hx(g) + hx(b)

fn render_scene(scene: Scene, backend: BackendKind, path: String):
    var f = open(path, String("w"))
    if f.is_null():
        return

    var theme = theme_scientific()
    var w = scene.width
    var h = scene.height
    var pad = scene.padding

    f.writeline(String("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"") + String(w) + String("\" height=\"") + String(h) + String("\">"))
    # background
    f.writeline(String("<rect x=\"0\" y=\"0\" width=\"") + String(w) + String("\" height=\"") + String(h) + String("\" fill=\"") + theme.background + String("\"/>"))

    # gridlines (vertical)
    var i = 0
    while i < len(scene.x_axis.ticks):
        let t = scene.x_axis.ticks[i]
        f.writeline(String("<line x1=\"") + String(t.pos) + String("\" y1=\"") + String(pad) +
                    String("\" x2=\"") + String(t.pos) + String("\" y2=\"") + String(h - pad) +
                    String("\" stroke=\"#e5e5e5\" stroke-width=\"1\" opacity=\"0.6\"/>"))
        i += 1
    # gridlines (horizontal)
    i = 0
    while i < len(scene.y_axis.ticks):
        let t = scene.y_axis.ticks[i]
        f.writeline(String("<line x1=\"") + String(pad) + String("\" y1=\"") + String(t.pos) +
                    String("\" x2=\"") + String(w - pad) + String("\" y2=\"") + String(t.pos) +
                    String("\" stroke=\"#e5e5e5\" stroke-width=\"1\" opacity=\"0.6\"/>"))
        i += 1

    # axes
    let x0 = pad
    let y0 = h - pad
    let x1 = w - pad
    let y1 = pad
    f.writeline(String("<line x1=\"") + String(x0) + String("\" y1=\"") + String(y0) +
                String("\" x2=\"") + String(x1) + String("\" y2=\"") + String(y0) +
                String("\" stroke=\"#222\" stroke-width=\"1.5\"/>"))
    f.writeline(String("<line x1=\"") + String(x0) + String("\" y1=\"") + String(y0) +
                String("\" x2=\"") + String(x0) + String("\" y2=\"") + String(y1) +
                String("\" stroke=\"#222\" stroke-width=\"1.5\"/>"))

    # ticks + labels
    i = 0
    while i < len(scene.x_axis.ticks):
        let t = scene.x_axis.ticks[i]
        f.writeline(String("<line x1=\"") + String(t.pos) + String("\" y1=\"") + String(y0) +
                    String("\" x2=\"") + String(t.pos) + String("\" y2=\"") + String(y0 + 6) +
                    String("\" stroke=\"#222\"/>"))
        f.writeline(String("<text x=\"") + String(t.pos) + String("\" y=\"") + String(y0 + 18) +
                    String("\" font-size=\"10\" text-anchor=\"middle\" fill=\"#111\">") + t.label + String("</text>"))
        i += 1
    i = 0
    while i < len(scene.y_axis.ticks):
        let t = scene.y_axis.ticks[i]
        f.writeline(String("<line x1=\"") + String(x0) + String("\" y1=\"") + String(t.pos) +
                    String("\" x2=\"") + String(x0 - 6) + String("\" y2=\"") + String(t.pos) +
                    String("\" stroke=\"#222\"/>"))
        f.writeline(String("<text x=\"") + String(x0 - 8) + String("\" y=\"") + String(t.pos + 3) +
                    String("\" font-size=\"10\" text-anchor=\"end\" fill=\"#111\">") + t.label + String("</text>"))
        i += 1

    # axis titles
    f.writeline(String("<text x=\"") + String((x0 + x1) * 0.5) + String("\" y=\"") + String(h - 8) +
                String("\" font-size=\"12\" text-anchor=\"middle\" fill=\"#111\">") + scene.x_title + String("</text>"))
    f.writeline(String("<text transform=\"translate(") + String(14) + String(",") + String((y1 + y0) * 0.5) + String(") rotate(-90)\"") +
                String(" font-size=\"12\" text-anchor=\"middle\" fill=\"#111\">") + scene.y_title + String("</text>"))

    # marks
    i = 0
    while i < len(scene.marks):
        var m = scene.marks[i]
        if m.kind.value == SceneMarkKinds.point().value:
            var j = 0
            while j < len(m.points):
                var p = m.points[j]
                var circ = String("<circle cx=\"") + String(p.x) + String("\" cy=\"") + String(p.y) +
                           String("\" r=\"") + String(m.size) + String("\" fill=\"") + _color_hex(m.color) + String("\"/>")
                f.writeline(circ)
                j += 1
        elif m.kind.value == SceneMarkKinds.line().value:
            var s = String("<polyline fill=\"none\" stroke=\"") + _color_hex(m.color) + String("\" stroke-width=\"") + String(m.size) + String("\" points=\"")
            var j = 0
            while j < len(m.points):
                var p = m.points[j]
                s += String(String(p.x) + String(",") + String(p.y))
                if j + 1 < len(m.points): s += String(" ")
                j += 1
            s += String("\"/>")
            f.writeline(s)
        elif m.kind.value == SceneMarkKinds.rect().value:
            var j = 0
            while j < len(m.points):
                var p = m.points[j]
                var wbar = 10.0
                var hbar = 10.0
                var x = p.x - wbar * 0.5
                var y = p.y - hbar * 0.5
                var rect = String("<rect x=\"") + String(x) + String("\" y=\"") + String(y) +
                            String("\" width=\"") + String(wbar) + String("\" height=\"") + String(hbar) +
                            String("\" fill=\"") + _color_hex(m.color) + String("\"/>")
                f.writeline(rect)
                j += 1
        i += 1

    # categorical legend (right)
    var lx = w - pad + 10
    var ly = pad
    i = 0
    while i < len(scene.legend):
        let item = scene.legend[i]
        let y = ly + i * 18
        f.writeline(String("<rect x=\"") + String(lx) + String("\" y=\"") + String(y-10) + String("\" width=\"12\" height=\"12\" fill=\"") + _color_hex(item.color) + String("\"/>"))
        f.writeline(String("<text x=\"") + String(lx + 18) + String("\" y=\"") + String(y) + String("\" font-size=\"11\" fill=\"#111\">") + item.label + String("</text>"))
        i += 1

    # continuous legend (vertical gradient)
    if scene.legend_cont.show:
        var gx = w - pad + 10
        var gy = h - pad - 120
        var gh = 100
        var gw = 14
        # approximate gradient: draw 20 stacked rects
        var k = 0
        while k < 20:
            var t = Float64(k) / 19.0
            var r = Int(68.0 + (220.0 - 68.0) * t)
            var g = Int(1.0 + (220.0 - 1.0) * t)
            var b = Int(84.0 + (47.0 - 84.0) * t)
            var y = gy + Int(Float64(gh) * (1.0 - t))
            f.writeline(String("<rect x=\"") + String(gx) + String("\" y=\"") + String(y) + String("\" width=\"") + String(gw) + String("\" height=\"5\" fill=\"#") +
                String("") + String("") + String("") + String("\" fill=\"") + String("#") + String("") + String("\"/>"))
            # We cannot embed hex via format easily; draw as colored rects using RGB approximations
            var hx = String("#")
            # simple hex write
            fn hx2(v: Int) -> String:
                var digits = String("0123456789ABCDEF")
                var hi = (v / 16) % 16
                var lo = v % 16
                return String("") + digits[hi:hi+1] + digits[lo:lo+1]
            var color = String("#") + hx2(r) + hx2(g) + hx2(b)
            f.writeline(String("<rect x=\"") + String(gx) + String("\" y=\"") + String(y) + String("\" width=\"") + String(gw) + String("\" height=\"5\" fill=\"") + color + String("\"/>"))
            k += 1
        f.writeline(String("<text x=\"") + String(gx + gw + 6) + String("\" y=\"") + String(gy + gh) + String("\" font-size=\"10\" fill=\"#111\">") + String(scene.legend_cont.min_val) + String("</text>"))
        f.writeline(String("<text x=\"") + String(gx + gw + 6) + String("\" y=\"") + String(gy) + String("\" font-size=\"10\" fill=\"#111\">") + String(scene.legend_cont.max_val) + String("</text>"))

    f.writeline(String("</svg>"))
    f.close()
