# ============================================================================
#  Momijo Visualization - spec/exporters/plotly_json.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.spec.spec import Spec, MarkKinds

fn to_plotly(spec: Spec) -> String:
    var out = String("{\"data\":[")
    # base trace
    var mode = String("markers")
    if spec.mark.value == MarkKinds.line().value: mode = String("lines")
    var x = spec.enc.x.field; var y = spec.enc.y.field
    out += String("{\"type\":\"scatter\",\"mode\":\"") + mode + String("\",\"x\":[")
    var n = len(spec.data.cols_num[x])
    var i = 0
    while i < n:
        out += String(spec.data.cols_num[x][i])
        if i + 1 < n: out += String(",")
        i += 1
    out += String("],\"y\":[")
    i = 0
    while i < n:
        out += String(spec.data.cols_num[y][i])
        if i + 1 < n: out += String(",")
        i += 1
    out += String("]}")
    # extra layers as traces
    var li = 0
    while li < len(spec.layers):
        let L = spec.layers[li]
        var mode2 = String("markers")
        if L.mark.value == MarkKinds.line().value: mode2 = String("lines")
        out += String(",{\"type\":\"scatter\",\"mode\":\"") + mode2 + String("\",\"x\":[")
        var n2 = len(spec.data.cols_num[L.enc.x.field]); var j = 0
        while j < n2:
            out += String(spec.data.cols_num[L.enc.x.field][j])
            if j + 1 < n2: out += String(",")
            j += 1
        out += String("],\"y\":[")
        j = 0
        while j < n2:
            out += String(spec.data.cols_num[L.enc.y.field][j])
            if j + 1 < n2: out += String(",")
            j += 1
        out += String("]}")
        li += 1
    out += String("]}")
    return out
