# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/funnel.mojo
# Description: Funnel chart builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_funnel_option(labels: List[String], values: List[Float64], title: String = "Funnel") -> String:
    var items = List[String](); var n = len(labels); var i = 0
    while i < n and i < len(values):
        items.append(String("{name:")+json_string(labels[i])+String(",value:")+String(values[i])+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"funnel\",data:")+json_array_str(items)+String("}]}")
    return opt