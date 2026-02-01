# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_polar.mojo
# Description: Polar charts

fn _pairs(theta: List[Float64], r: List[Float64]) -> String:
    var n=len(theta); var data="["; var i=0
    while i<n: data+="["+String(theta[i])+","+String(r[i])+"]"; if i+1<n: data+=","; i+=1
    data+="]"; return data

fn chart_polar_line(title: String, theta: List[Float64], r: List[Float64]) -> String:
    var data=_pairs(theta,r)
    return "{\"title\":{\"text\":\""+title+"\"},\"polar\":{},\"angleAxis\":{},\"radiusAxis\":{},\"series\":[{\"coordinateSystem\":\"polar\",\"type\":\"line\",\"data\":"+data+"}]}"

fn chart_polar_scatter(title: String, theta: List[Float64], r: List[Float64]) -> String:
    var data=_pairs(theta,r)
    return "{\"title\":{\"text\":\""+title+"\"},\"polar\":{},\"angleAxis\":{},\"radiusAxis\":{},\"series\":[{\"coordinateSystem\":\"polar\",\"type\":\"scatter\",\"data\":"+data+"}]}"

fn chart_polar_bar(title: String, theta: List[Float64], r: List[Float64]) -> String:
    var data=_pairs(theta,r)
    return "{\"title\":{\"text\":\""+title+"\"},\"polar\":{},\"angleAxis\":{},\"radiusAxis\":{},\"series\":[{\"coordinateSystem\":\"polar\",\"type\":\"bar\",\"data\":"+data+"}]}"