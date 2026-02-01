# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/map.mojo
# Description: Map builder (world).

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_map_option(pairs: List[(String, Float64)], title: String = "World Map Choropleth") -> String:
    var data = List[String](); var i = 0
    while i < len(pairs):
        var (name, val) = pairs[i]
        data.append(String("{name:")+json_string(name)+String(",value:")+String(val)+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},visualMap:{min:0,max:100,left:'left',top:'bottom'},")
    opt += String("series:[{type:'map',map:'world',roam:true,data:")+json_array_str(data)+String("}]}")
    return opt