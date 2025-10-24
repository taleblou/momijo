# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/parallel.mojo
# Description: Parallel coordinates builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_parallel_option(dim_names: List[String], rows: List[List[Float64]], title: String = "Parallel") -> String:
    var dims = List[String](); var i = 0
    while i < len(dim_names):
        dims.append(String("{name:")+json_string(dim_names[i])+String("}")); i += 1
    var data = List[String](); var r = 0
    while r < len(rows):
        var parts = List[String](); var c = 0
        while c < len(rows[r]): parts.append(String(rows[r][c])); c += 1
        data.append(json_array_str(parts)); r += 1
    var opt = String("{title:{text:")+json_string(title)+String("},parallelAxis:")+json_array_str(dims)+String(",series:[{type:\"parallel\",data:")+json_array_str(data)+String("}]}")
    return opt