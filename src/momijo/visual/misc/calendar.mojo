# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/calendar.mojo
# Description: Calendar heatmap builder.

from momijo.visual.echarts.core import json_string, json_array_str

# items: List of (date YYYY-MM-DD, value)
fn echarts_calendar_option(range_str: String, items: List[(String, Float64)], title: String = "Calendar Heatmap") -> String:
    var data = List[String](); var i = 0
    while i < len(items):
        var (d, v) = items[i]
        data.append(String("[")+json_string(d)+String(",")+String(v)+String("]")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},calendar:{range:")+json_string(range_str)+String("},")
    opt += String("visualMap:{min:0,max:100,orient:\"horizontal\",left:\"center\"},")
    opt += String("series:[{type:\"heatmap\",coordinateSystem:\"calendar\",data:")+json_array_str(data)+String("}]}")
    return opt