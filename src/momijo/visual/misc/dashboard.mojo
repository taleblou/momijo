# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/dashboard.mojo

fn echarts_dashboard_html(options: List[String], titles: List[String], layout: String = "grid") -> String:
    var html = String("<!doctype html><html><head><meta charset='utf-8'>")
    html += String("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    html += String("<script src='https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js'></script>")
    html += String("<style>body{background:#0b0f19;color:#eee;font-family:system-ui;margin:0;padding:24px;}")
    html += String(".chart{height:420px;background:#121826;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.35);}")
    html += String(".title{margin:6px 0 12px 0;font-weight:600;}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px;}")
    html += String("</style></head><body><div class='grid'>")
    var i = 0
    while i < len(options):
        html += String("<div><div class='title'>")+titles[i]+String("</div><div id='c")+String(i)+String("' class='chart'></div></div>")
        i += 1
    html += String("</div><script>var charts=[];")
    var k = 0
    while k < len(options):
        html += String("charts.push(echarts.init(document.getElementById('c")+String(k)+String("'))); charts[")+String(k)+String("].setOption(")+options[k]+String(");")
        k += 1
    html += String("window.addEventListener('resize',()=>charts.forEach(c=>c.resize()));</script></body></html>")
    return html