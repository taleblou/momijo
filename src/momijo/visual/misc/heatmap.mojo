# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/heatmap.mojo
# Description: Heatmap builder.

from momijo.visual.echarts.core import json_string, json_string_array, json_array_str

fn echarts_heatmap_option(x_labels: List[String], y_labels: List[String], values: List[List[Float64]],
                          title: String = "Heatmap") -> String:
    var data = List[String](); var yi = 0
    while yi < len(y_labels) and yi < len(values):
        var row = values[yi]; var xi = 0
        while xi < len(x_labels) and xi < len(row):
            data.append(String("[")+String(xi)+String(",")+String(yi)+String(",")+String(row[xi])+String("]")); xi += 1
        yi += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{position:\"top\"},grid:{left:80,top:60},")
    opt += String("xAxis:{type:\"category\",data:")+json_string_array(x_labels)+String(",splitArea:{show:true}},")
    opt += String("yAxis:{type:\"category\",data:")+json_string_array(y_labels)+String(",splitArea:{show:true}},")
    opt += String("visualMap:{min:0,max:100,calculable:true,orient:\"horizontal\",left:\"center\",bottom:10},")
    opt += String("series:[{type:\"heatmap\",data:")+json_array_str(data)+String(",label:{show:false}}]}")
    return opt