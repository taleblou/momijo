# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/candlestick.mojo
# Description: Candlestick builder.

from momijo.visual.echarts.core import json_string, json_string_array, json_array_str

fn echarts_candlestick_option(x: List[String], o: List[Float64], h: List[Float64], l: List[Float64], c: List[Float64],
                              title: String = "Candlestick") -> String:
    var x_json = json_string_array(x)
    var n = len(o); var pts = List[String](); var i = 0
    while i < n and i < len(h) and i < len(l) and i < len(c):
        pts.append(String("[")+String(o[i])+String(",")+String(c[i])+String(",")+String(l[i])+String(",")+String(h[i])+String("]")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{trigger:\"axis\"},xAxis:{type:\"category\",data:")+x_json+String("},yAxis:{scale:true},series:[{type:\"candlestick\",data:")+json_array_str(pts)+String("}]}")
    return opt