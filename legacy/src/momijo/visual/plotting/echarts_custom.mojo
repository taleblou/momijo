# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_custom.mojo
# Description: custom helpers

fn chart_stem(title: String, xs: List[Float64], ys: List[Float64], baseline: Float64=0.0) -> String:
    var data="["; var i=0
    while i<len(xs):
        data+="{\"x\":"+String(xs[i])+",\"y\":"+String(ys[i])+"}"
        if i+1<len(xs): data+=","
        i+=1
    data+="]"
    var render="function(p,api){var x=api.value('x'),y=api.value('y');var xs=api.coord([x,0])[0];var y0=api.coord([x,"+String(baseline)+"])[1];var y1=api.coord([x,y])[1];return {type:'line',shape:{x1:xs,y1:y0,x2:xs,y2:y1},style:api.style({})};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"

fn chart_errorbar(title: String, xs: List[Float64], ys: List[Float64], yerr: List[Float64]) -> String:
    var data="["; var i=0
    while i<len(xs):
        var ylo=ys[i]-yerr[i]; var yhi=ys[i]+yerr[i]
        data+="{\"x\":"+String(xs[i])+",\"y\":"+String(ys[i])+",\"ylo\":"+String(ylo)+",\"yhi\":"+String(yhi)+"}"
        if i+1<len(xs): data+=","
        i+=1
    data+="]"
    var render="function(p,api){var x=api.value('x'),y=api.value('y'),ylo=api.value('ylo'),yhi=api.value('yhi');var cx=api.coord([x,y])[0];var y1=api.coord([x,ylo])[1];var y2=api.coord([x,yhi])[1];var cap=6;return {type:'group',children:[{type:'line',shape:{x1:cx,y1:y1,x2:cx,y2:y2},style:api.style({})},{type:'line',shape:{x1:cx-cap,y1:y1,x2:cx+cap,y2:y1},style:api.style({})},{type:'line',shape:{x1:cx-cap,y1:y2,x2:cx+cap,y2:y2},style:api.style({})}]};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"line\",\"data\":[]},"+s+"]}"

fn chart_violin(title: String, categories: List[String], xs: List[List[Float64]], ys: List[List[Float64]]) -> String:
    var polys="["; var i=0
    while i<len(categories):
        polys+="{\"name\":\""+categories[i]+"\",\"points\":["
        var j=0
        while j<len(xs[i]):
            polys+="["+String(xs[i][j])+","+String(ys[i][j])+"]"
            if j+1<len(xs[i]): polys+=","
            j+=1
        j=len(xs[i])-1
        while j>=0:
            polys+=",["+String(-xs[i][j])+","+String(ys[i][j])+"]"
            j-=1
        polys+="]}"
        if i+1<len(categories): polys+=","
        i+=1
    polys+="]"
    var render="function(p,api){var pts=api.value('points'),pp=[];for(var i=0;i<pts.length;i++){pp.push(api.coord(pts[i]));}return {type:'polygon',shape:{points:pp},style:api.style({fill:'#aaa'})};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+polys+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"

fn chart_eventplot(title: String, rows: Int, events: List[List[Float64]]) -> String:
    var data="["; var i=0
    while i<rows:
        var j=0
        while j<len(events[i]):
            data+="{\"x\":"+String(events[i][j])+",\"y\":"+String(i)+"}"
            if not (i==rows-1 and j==len(events[i])-1): data+=","
            j+=1
        i+=1
    data+="]"
    var render="function(p,api){var x=api.value('x'),y=api.value('y');var p1=api.coord([x,y-0.4]),p2=api.coord([x,y+0.4]);return {type:'line',shape:{x1:p1[0],y1:p1[1],x2:p2[0],y2:p2[1]},style:api.style({})};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"

fn chart_contour_lines(title: String, lines: List[List[List[Float64]]]) -> String:
    var data="["; var i=0
    while i<len(lines):
        data+="{\"pts\":["
        var j=0
        while j<len(lines[i]):
            data+="["+String(lines[i][j][0])+","+String(lines[i][j][1])+"]"
            if j+1<len(lines[i]): data+=","
            j+=1
        data+="]}"
        if i+1<len(lines): data+=","
        i+=1
    data+="]"
    var render="function(p,api){var pts=api.value('pts'),pp=[];for(var i=0;i<pts.length;i++){pp.push(api.coord(pts[i]));}return {type:'polyline',shape:{points:pp},style:api.style({})};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"

fn chart_hexbin(title: String, centers: List[List[Float64]], size: Float64, values: List[Float64], vmin: Float64, vmax: Float64) -> String:
    var data="["; var i=0
    while i<len(centers):
        data+="{\"x\":"+String(centers[i][0])+",\"y\":"+String(centers[i][1])+",\"v\":"+String(values[i])+"}"
        if i+1<len(centers): data+=","
        i+=1
    data+="]"
    var render="function(p,api){var x=api.value('x'),y=api.value('y'),v=api.value('v');var R="+String(size)+";var pts=[];for(var k=0;k<6;k++){var a=Math.PI/3*k;pts.push(api.coord([x+R*Math.cos(a),y+R*Math.sin(a)]));}var t=echarts.number.linearMap(v, ["+String(vmin)+","+String(vmax)+"], [0,1], true);return {type:'polygon',shape:{points:pts},style:{fill:echarts.color.modifyHSL('#4477ee',0,0,t*100),stroke:'#999'}};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"

fn chart_quiver(title: String, xs: List[Float64], ys: List[Float64], u: List[Float64], v: List[Float64], scale: Float64=1.0) -> String:
    var data="["; var i=0
    while i<len(xs):
        data+="{\"x\":"+String(xs[i])+",\"y\":"+String(ys[i])+",\"u\":"+String(u[i])+",\"v\":"+String(v[i])+"}"
        if i+1<len(xs): data+=","
        i+=1
    data+="]"
    var render="function(p,api){var x=api.value('x'),y=api.value('y'),u=api.value('u'),v=api.value('v');var p0=api.coord([x,y]);var p1=api.coord([x+u*"+String(scale)+",y+v*"+String(scale)+"]);return {type:'group',children:[{type:'line',shape:{x1:p0[0],y1:p0[1],x2:p1[0],y2:p1[1]},style:api.style({})}]};}"
    var s="{\"type\":\"custom\",\"renderItem\":"+render+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"