# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/pictorialbar.mojo
# Description: PictorialBar builder.

from momijo.visual.echarts.core import json_string, json_string_array, json_array_str

fn echarts_pictorialbar_option(x: List[String], values: List[Float64], title: String = "PictorialBar", symbol: String = "rect") -> String:
    var x_json = json_string_array(x)
    var vs = List[String](); var i = 0
    while i < len(values): vs.append(String(values[i])); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},xAxis:{type:\"category\",data:")+x_json+String("},yAxis:{type:\"value\"},")
    opt += String("series:[{type:\"pictorialBar\",symbol:")+json_string(symbol)+String(",symbolRepeat:true,data:")+json_array_str(vs)+String("}]}")
    return opt