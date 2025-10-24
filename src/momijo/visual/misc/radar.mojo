# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/radar.mojo
# Description: Radar chart builder.

from momijo.visual.echarts.core import json_string, json_array_str, json_number_array

fn echarts_radar_option(indicators: List[(String, Float64)], series: List[List[Float64]], series_names: List[String],
                        title: String = "Radar") -> String:
    var inds = List[String](); var i = 0
    while i < len(indicators):
        var (name, maxv) = indicators[i]
        inds.append(String("{name:")+json_string(name)+String(",max:")+String(maxv)+String("}")); i += 1
    var sers = List[String](); i = 0
    while i < len(series):
        var data = json_number_array(series[i])
        var nm = json_string(series_names[i]) if i < len(series_names) else json_string(String("series ")+String(i))
        sers.append(String("{name:")+nm+String(",value:")+data+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},legend:{},radar:{indicator:")+json_array_str(inds)+String("},")
    opt += String("series:[{type:\"radar\",data:")+json_array_str(sers)+String("}]}")
    return opt