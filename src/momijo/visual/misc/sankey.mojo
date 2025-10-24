# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/sankey.mojo
# Description: Sankey builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_sankey_option(nodes: List[String], links: List[(String,String,Float64)], title: String = "Sankey") -> String:
    var ns = List[String](); var i = 0
    while i < len(nodes):
        ns.append(String("{name:")+json_string(nodes[i])+String("}")); i += 1
    var ls = List[String](); i = 0
    while i < len(links):
        var (s, t, v) = links[i]
        ls.append(String("{source:")+json_string(s)+String(",target:")+json_string(t)+String(",value:")+String(v)+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"sankey\",emphasis:{focus:\"adjacency\"},data:")+json_array_str(ns)+String(",links:")+json_array_str(ls)+String("}]}")
    return opt