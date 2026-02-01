# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/treemap.mojo
# Description: Treemap builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_treemap_option(names: List[String], values: List[Float64], title: String = "Treemap") -> String:
    var items = List[String](); var i = 0
    while i < len(names) and i < len(values):
        items.append(String("{name:")+json_string(names[i])+String(",value:")+String(values[i])+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"treemap\",data:")+json_array_str(items)+String("}]}")
    return opt