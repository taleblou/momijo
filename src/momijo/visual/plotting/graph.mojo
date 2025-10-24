# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/graph.mojo
# Description: Graph builder.

from momijo.visual.echarts.core import json_string, json_array_str

fn echarts_graph_option(nodes: List[String], links: List[(Int,Int)], title: String = "Graph") -> String:
    var ns = List[String](); var i = 0
    while i < len(nodes): ns.append(String("{name:")+json_string(nodes[i])+String("}")); i += 1
    var ls = List[String](); i = 0
    while i < len(links):
        var (s, t) = links[i]
        ls.append(String("{source:")+String(s)+String(",target:")+String(t)+String("}")); i += 1
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"graph\",layout:\"force\",roam:true,data:")+json_array_str(ns)+String(",links:")+json_array_str(ls)+String(",force:{repulsion:200}}]}")
    return opt