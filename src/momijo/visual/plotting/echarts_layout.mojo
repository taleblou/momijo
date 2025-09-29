# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_layout.mojo
# Description: Layout helpers

fn visualMap_continuous(minv: Float64, maxv: Float64, orient: String="horizontal") -> String:
    return "{\"type\":\"continuous\",\"min\":"+String(minv)+",\"max\":"+String(maxv)+",\"orient\":\""+orient+"\"}"

fn visualMap_piecewise(pieces: String, orient: String="horizontal") -> String:
    return "{\"type\":\"piecewise\",\"pieces\":"+pieces+",\"orient\":\""+orient+"\"}"

fn timeline_builder(labels: List[String]) -> String:
    var s="["; var i=0
    while i<len(labels): s+="\""+labels[i]+"\""; if i+1<len(labels): s+=","; i+=1
    s+="]"
    return "{\"axisType\":\"category\",\"data\":"+s+"}"

fn dataset_builder(source: String) -> String:
    return "{\"source\":"+source+"}"