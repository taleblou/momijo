# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/sunburst.mojo
# Description: Sunburst builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_sunburst_option(labels: List[String], values: List[Float64], title: String = "Sunburst") -> String:
    var items = List[String](); var i = 0
    while i < len(labels) and i < len(values):
        items.append(String("{name:")+json_string(labels[i])+String(",value:")+String(values[i])+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"sunburst\",radius:[\"20%\",\"80%\"],data:")+json_array_str(items)+String("}]}")
    return opt