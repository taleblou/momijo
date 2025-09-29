# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/themeriver.mojo
# Description: ThemeRiver builder.

from momijo.visual.echarts.core import json_string, json_array_str

# dates: List[String], names: List[String], matrix values: [len(dates) x len(names)]
fn echarts_themeriver_option(dates: List[String], names: List[String], matrix: List[List[Float64]], title: String = "ThemeRiver") -> String:
    var data = List[String](); var di = 0
    while di < len(dates) and di < len(matrix):
        var row = matrix[di]; var ni = 0
        while ni < len(names) and ni < len(row):
            data.append(String("[")+json_string(dates[di])+String(",")+String(row[ni])+String(",")+json_string(names[ni])+String("]"))
            ni += 1
        di += 1
    var opt = String("{title:{text:")+json_string(title)+String("},legend:{data:")+json_array_str([json_string(n) for n in names])+String("},singleAxis:{type:\"time\"},")
    opt += String("series:[{type:\"themeRiver\",data:")+json_array_str(data)+String("}]}")
    return opt