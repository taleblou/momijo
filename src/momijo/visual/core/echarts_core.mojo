# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_core.mojo
# Description: Core JSON utils

fn _join(xs: List[String]) -> String:
    var s=String(""); var i=0
    while i<len(xs): s+=xs[i]; if i+1<len(xs): s+=","; i+=1
    return s

fn json_array_f64(xs: List[Float64]) -> String:
    var parts=List[String](); var i=0
    while i<len(xs): parts.append(String(xs[i])); i+=1
    return "[" + _join(parts) + "]"

fn json_array_str(xs: List[String]) -> String:
    var parts=List[String](); var i=0
    while i<len(xs): parts.append("\""+xs[i]+"\""); i+=1
    return "[" + _join(parts) + "]"