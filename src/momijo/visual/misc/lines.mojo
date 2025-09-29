# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/lines.mojo
# Description: Geo Lines builder (requires world map).

from momijo.visual.echarts.core import json_string, json_array_str

# lines: list of ((lng1,lat1), (lng2,lat2))
fn echarts_lines_option(lines: List[((Float64,Float64),(Float64,Float64))], title: String = "Geo Lines") -> String:
    var data = List[String](); var i = 0
    while i < len(lines):
        var (a,b) = lines[i]
        var (x1,y1) = a; var (x2,y2) = b
        data.append(String("{coords:[[")+String(x1)+String(",")+String(y1)+String("],[")+String(x2)+String(",")+String(y2)+String("]]}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},geo:{map:'world',roam:true},series:[{type:'lines',coordinateSystem:'geo',data:")+json_array_str(data)+String("}]}")
    return opt