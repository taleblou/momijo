# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_3d.mojo
# Description: 3D charts

fn _wrap3d(title: String, series: String) -> String:
    return "{\"title\":{\"text\":\""+title+"\"},\"tooltip\":{\"show\":true},\"grid3D\":{},\"xAxis3D\":{\"type\":\"value\"},\"yAxis3D\":{\"type\":\"value\"},\"zAxis3D\":{\"type\":\"value\"},\"series\":["+series+"]}"

fn chart_surface3d(title: String, rows: Int, cols: Int, zvals: List[Float64]) -> String:
    var data="["; var r=0
    while r<rows:
        var c=0
        while c<cols:
            var idx=r*cols+c
            data+="["+String(c)+","+String(r)+","+String(zvals[idx])+"]"
            if not (r==rows-1 and c==cols-1): data+=","
            c+=1
        r+=1
    data+="]"
    var s="{\"type\":\"surface\",\"data\":"+data+",\"shading\":\"lambert\"}"
    return _wrap3d(title,s)

fn chart_line3d(title: String, path: List[List[Float64]]) -> String:
    var data="["; var i=0
    while i<len(path):
        data+="["+String(path[i][0])+","+String(path[i][1])+","+String(path[i][2])+"]"
        if i+1<len(path): data+=","
        i+=1
    data+="]"
    var s="{\"type\":\"line3D\",\"data\":"+data+",\"lineStyle\":{\"width\":2}}"
    return _wrap3d(title,s)

fn chart_scatter3d(title: String, points: List[List[Float64]]) -> String:
    var data="["; var i=0
    while i<len(points):
        data+="["+String(points[i][0])+","+String(points[i][1])+","+String(points[i][2])+"]"
        if i+1<len(points): data+=","
        i+=1
    data+="]"
    var s="{\"type\":\"scatter3D\",\"data\":"+data+"}"
    return _wrap3d(title,s)

fn chart_bar3d(title: String, bars: List[List[Float64]]) -> String:
    var data="["; var i=0
    while i<len(bars):
        data+="["+String(bars[i][0])+","+String(bars[i][1])+","+String(bars[i][2])+"]"
        if i+1<len(bars): data+=","
        i+=1
    data+="]"
    var s="{\"type\":\"bar3D\",\"data\":"+data+",\"shading\":\"lambert\"}"
    return _wrap3d(title,s)

fn chart_lines3d(title: String, segments: List[List[List[Float64]]]) -> String:
    var data="["; var i=0
    while i<len(segments):
        data+="{\"coords\":["
        var j=0
        while j<len(segments[i]):
            data+="["+String(segments[i][j][0])+","+String(segments[i][j][1])+","+String(segments[i][j][2])+"]"
            if j+1<len(segments[i]): data+=","
            j+=1
        data+="]}"
        if i+1<len(segments): data+=","
        i+=1
    data+="]"
    var s="{\"type\":\"lines3D\",\"data\":"+data+",\"lineStyle\":{\"width\":2}}"
    return _wrap3d(title,s)

fn chart_globe_with_textures(title: String, baseTextureUrl: String, heightTextureUrl: String) -> String:
    return "{\"title\":{\"text\":\""+title+"\"},\"globe\":{\"baseTexture\":\""+baseTextureUrl+"\",\"heightTexture\":\""+heightTextureUrl+"\",\"shading\":\"lambert\",\"viewControl\":{\"autoRotate\":true}},\"series\":[]}"

fn chart_geo3d_world(title: String, map_name: String, geojson_url: String) -> String:
    return "{\"title\":{\"text\":\""+title+"\"},\"__registerMap__\":{\"name\":\""+map_name+"\",\"url\":\""+geojson_url+"\"},\"geo3D\":{\"map\":\""+map_name+"\",\"boxHeight\":10,\"regionHeight\":3},\"series\":[{\"type\":\"map3D\",\"map\":\""+map_name+"\"}]}"