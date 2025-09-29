# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/pie.mojo
# Description: Pie chart builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_pie_option(labels: List[String], values: List[Float64], title: String = "Pie") -> String:
    var items = List[String](); var n = len(labels); var i = 0
    while i < n and i < len(values):
        items.append(String("{name:")+json_string(labels[i])+String(",value:")+String(values[i])+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{trigger:\"item\"},legend:{},")
    opt += String("series:[{type:\"pie\",radius:[\"30%\",\"70%\"],avoidLabelOverlap:true,data:")+json_array_str(items)+String("}]}")
    return opt