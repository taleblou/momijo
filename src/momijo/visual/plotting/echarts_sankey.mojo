# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_sankey.mojo
# Description: Sankey

fn _nodes(names: List[String]) -> String:
    var s="["; var i=0
    while i<len(names): s+="{\"name\":\""+names[i]+"\"}"; if i+1<len(names): s+=","; i+=1
    s+="]"; return s

fn _links(pairs: List[List[String]]) -> String:
    var s="["; var i=0
    while i<len(pairs):
        s+="{\"source\":\""+pairs[i][0]+"\",\"target\":\""+pairs[i][1]+"\",\"value\":1}"
        if i+1<len(pairs): s+=","
        i+=1
    s+="]"; return s

fn chart_sankey(title: String, nodes: List[String], links: List[List[String]]) -> String:
    var ns=_nodes(nodes); var ls=_links(links)
    return "{\"title\":{\"text\":\""+title+"\"},\"series\":[{\"type\":\"sankey\",\"data\":"+ns+",\"links\":"+ls+"}]}"