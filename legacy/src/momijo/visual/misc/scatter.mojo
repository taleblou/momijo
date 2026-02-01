# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/scatter.mojo
# Description: Scatter chart builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_scatter_option(x: List[Float64], y: List[Float64],
                          title: String = "Scatter", x_name: String = "x", y_name: String = "y") -> String:
    var pts = List[String](); var n = len(x); var i = 0
    while i < n and i < len(y):
        pts.append(String("[")+String(x[i])+String(",")+String(y[i])+String("]")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},")
    opt += String("xAxis:{name:")+json_string(x_name)+String("},yAxis:{name:")+json_string(y_name)+String("},")
    opt += String("series:[{type:\"scatter\",symbolSize:8,data:")+json_array_str(pts)+String("}]}")
    return opt