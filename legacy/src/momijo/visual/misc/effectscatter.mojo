# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/effectscatter.mojo
# Description: EffectScatter builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_effectscatter_option(points: List[(Float64, Float64)], title: String = "EffectScatter") -> String:
    var data = List[String](); var i = 0
    while i < len(points):
        var (x, y) = points[i]
        data.append(String("[")+String(x)+String(",")+String(y)+String("]")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},xAxis:{},yAxis:{},series:[{type:\"effectScatter\",rippleEffect:{scale:2.5},data:")+json_array_str(data)+String("}]}")
    return opt