# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/bar.mojo
# Description: Bar chart builder.

from momijo.visual.echarts.core import json_string, json_number_array, json_string_array, json_array_str

fn echarts_bar_option(x: List[String], y_series: List[List[Float64]], series_names: List[String],
                      stacked: Bool = False,
                      title: String = "Bar Chart", y_name: String = "value") -> String:
    var x_json = json_string_array(x)
    var series = List[String](); var i = 0
    while i < len(y_series):
        var data_json = json_number_array(y_series[i])
        var nm = json_string(series_names[i]) if i < len(series_names) else json_string(String("series ")+String(i))
        var s = String("{type:\"bar\",name:")+nm
        if stacked: s += String(",stack:\"total\"")
        s += String(",data:")+data_json+String("}")
        series.append(s); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{trigger:\"axis\"},legend:{},")
    opt += String("grid:{left:40,right:20,top:50,bottom:40},xAxis:{type:\"category\",data:")+x_json+String("},")
    opt += String("yAxis:{type:\"value\",name:")+json_string(y_name)+String("},series:")+json_array_str(series)+String("}")
    return opt