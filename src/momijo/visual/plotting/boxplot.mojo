# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/boxplot.mojo
# Description: Boxplot builder.

from momijo.visual.echarts.core import json_string, json_string_array, json_array_str

# values expects rows of [min, q1, median, q3, max] per category
fn echarts_boxplot_option(x_labels: List[String], values: List[List[Float64]], title: String = "Boxplot") -> String:
    var x_json = json_string_array(x_labels)
    var rows = List[String](); var i = 0
    while i < len(values):
        var r = values[i]; var parts = List[String](); var j = 0
        while j < len(r): parts.append(String(r[j])); j += 1
        rows.append(json_array_str(parts)); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{trigger:\"item\"},xAxis:{type:\"category\",data:")+x_json+String("},yAxis:{type:\"value\"},")
    opt += String("series:[{type:\"boxplot\",data:")+json_array_str(rows)+String("}]}")
    return opt