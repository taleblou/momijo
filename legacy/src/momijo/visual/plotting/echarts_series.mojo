# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_series.mojo
# Description: 2D charts incl boxplot

from momijo.visual.echarts_core import json_array_f64 as _f64
from momijo.visual.echarts_core import json_array_str as _s

fn _wrap(title: String, s: String, xA: String, yA: String, trigger: String="axis") -> String:
    return "{\"title\":{\"text\":\""+title+"\"},\"tooltip\":{\"trigger\":\""+trigger+"\"},\"legend\":{\"show\":true},\"grid\":{\"containLabel\":true},\"xAxis\":"+xA+",\"yAxis\":"+yA+",\"series\":["+s+"]}"

fn chart_line(title: String, xs: List[Float64], ys: List[Float64], step: Bool=False, area: Bool=False) -> String:
    var data="["; var i=0
    while i<len(xs):
        data+="["+String(xs[i])+","+String(ys[i])+"]"; if i+1<len(xs): data+=","; i+=1
    data+="]"
    var stepv=(if step then "\"step\":\"middle\"," else ""); var areav=(if area then "\"areaStyle\":{}," else "")
    var s="{\"type\":\"line\","+stepv+areav+"\"data\":"+data+"}"
    return _wrap(title,s,"{\"type\":\"value\"}","{\"type\":\"value\"}")

fn chart_bar(title: String, labels: List[String], values: List[Float64], horizontal: Bool=False, stacked_key: String="") -> String:
    var s="{\"type\":\"bar\""+(if len(stacked_key)>0 then ",\"stack\":\""+stacked_key+"\"" else "")+",\"data\":"+_f64(values)+"}"
    var xA=(if horizontal then "{\"type\":\"value\"}" else "{\"type\":\"category\",\"data\":"+_s(labels)+"}")
    var yA=(if horizontal then "{\"type\":\"category\",\"data\":"+_s(labels)+"}" else "{\"type\":\"value\"}")
    return _wrap(title,s,xA,yA)

fn chart_scatter(title: String, xs: List[Float64], ys: List[Float64], sizes: List[Float64]) -> String:
    var n=len(xs); var data="["; var i=0
    while i<n: data+="["+String(xs[i])+","+String(ys[i])+","+String(sizes[i])+"]"; if i+1<n: data+=","; i+=1
    data+="]"
    var s="{\"type\":\"scatter\",\"data\":"+data+",\"symbolSize\":(d)=>d[2]}"
    return _wrap(title,s,"{\"type\":\"value\"}","{\"type\":\"value\"}","item")

fn chart_pie(title: String, labels: List[String], values: List[Float64], donut: Bool=False) -> String:
    var radius=(if donut then "[\"40%\",\"70%\"]" else "\"70%\"")
    var data="["; var i=0
    while i<len(labels):
        data+="{\"name\":\""+labels[i]+"\",\"value\":"+String(values[i])+"}"; if i+1<len(labels): data+=","; i+=1
    data+="]"
    var s="{\"type\":\"pie\",\"radius\":"+radius+",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"tooltip\":{\"trigger\":\"item\"},\"legend\":{\"show\":true},\"series\":["+s+"]}"

fn chart_heatmap(title: String, rows: Int, cols: Int, values: List[Float64]) -> String:
    var data="["; var r=0
    while r<rows:
        var c=0
        while c<cols:
            var idx=r*cols+c
            data+="["+String(c)+","+String(r)+","+String(values[idx])+"]"
            if not (r==rows-1 and c==cols-1): data+=","
            c+=1
        r+=1
    data+="]"
    var s="{\"type\":\"heatmap\",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"visualMap\":{\"min\":0,\"max\":1,\"calculable\":true},\"series\":["+s+"]}"

# -------- Boxplot (compute 5-number summary) --------
fn _percentile(sorted_vals: List[Float64], p: Float64) -> Float64:
    var n=len(sorted_vals)
    if n==0: return 0.0
    var pos=(Float64(n)-1.0)*p
    var i=Int(pos); var frac=pos-Float64(i)
    if i+1<n: return sorted_vals[i]*(1.0-frac)+sorted_vals[i+1]*frac
    return sorted_vals[n-1]

fn chart_boxplot(title: String, groups: List[List[Float64]]) -> String:
    var data="["; var g=0
    while g<len(groups):
        # sort (simple selection sort)
        var xs=groups[g]; var i=0
        while i<len(xs):
            var j=i+1
            while j<len(xs):
                if xs[j] < xs[i]:
                    var t=xs[i]; xs[i]=xs[j]; xs[j]=t
                j+=1
            i+=1
        var minv = xs[0]; var maxv = xs[len(xs)-1]
        var q1 = _percentile(xs, 0.25); var med = _percentile(xs, 0.5); var q3 = _percentile(xs, 0.75)
        data+="["+String(minv)+","+String(q1)+","+String(med)+","+String(q3)+","+String(maxv)+"]"
        if g+1<len(groups): data+=","
        g+=1
    data+="]"
    var s="{\"type\":\"boxplot\",\"data\":"+data+"}"
    return "{\"title\":{\"text\":\""+title+"\"},\"xAxis\":{\"type\":\"category\",\"data\":"+_s(["G1","G2","G3","G4"])+"},\"yAxis\":{\"type\":\"value\"},\"series\":["+s+"]}"