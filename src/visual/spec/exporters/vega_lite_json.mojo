# ============================================================================
#  Momijo Visualization - spec/exporters/vega_lite_json.mojo
#  (c) 2025  MIT
#  NOTE: Mojo scaffolding for FFI/SIMD; adapt to your toolchain.
# ============================================================================

from momijo.visual.spec.spec import Spec, MarkKinds

fn _mark_name(kind_val: Int) -> String:
    if kind_val == MarkKinds.line().value: return String("line")
    if kind_val == MarkKinds.rect().value: return String("bar")
    return String("point")

fn _enc_xy_color(xf: String, yf: String, cf: String) -> String:
    var s = String("\"x\":{\"field\":\"") + xf + String("\",\"type\":\"quantitative\"},") + \
            String("\"y\":{\"field\":\"") + yf + String("\",\"type\":\"quantitative\"}")
    if len(cf) > 0:
        s += String(",\"color\":{\"field\":\"") + cf + String("\",\"type\":\"nominal\"}")
    return s

fn to_vega_lite(spec: Spec) -> String:
    var out = String("{\"$schema\":\"https://vega.github.io/schema/vega-lite/v5.json\",")
    # layers
    out += String("\"layer\":[")
    # base
    out += String("{\"mark\":\"") + _mark_name(spec.mark.value) + String("\",\"encoding\":{") + _enc_xy_color(spec.enc.x.field, spec.enc.y.field, spec.enc.color.field) + String("}}")
    # extras
    var i = 0
    while i < len(spec.layers):
        let L = spec.layers[i]
        out += String(",{\"mark\":\"") + _mark_name(L.mark.value) + String("\",\"encoding\":{") + _enc_xy_color(L.enc.x.field, L.enc.y.field, L.enc.color.field) + String("}}")
        i += 1
    out += String("],")

    # data inline (use only base fields for brevity)
    var x = spec.enc.x.field; var y = spec.enc.y.field
    out += String("\"data\":{\"values\":[")
    var n = len(spec.data.cols_num[x]); var j = 0
    while j < n:
        out += String("{\"") + x + String("\":") + String(spec.data.cols_num[x][j]) + String(",\"") + y + String("\":") + String(spec.data.cols_num[y][j]) + String("}")
        if j + 1 < n: out += String(",")
        j += 1
    out += String("]},")

    # facet support (row/column) + resolve policy
    if len(spec.facet.by) > 0:
        out += String("\"facet\":{")
        out += String("\"row\":{\"field\":\"") + spec.facet.by + String("\",\"type\":\"nominal\"}")
        out += String("},")
        out += String("\"resolve\":{\"scale\":{\"x\":\"independent\",\"y\":\"independent\"}},");

    out += String("\"config\":{\"view\":{\"stroke\":\"transparent\"}}")
    out += String("}")
    return out
