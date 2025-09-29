# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/core.mojo

fn _escape_json_str(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "\\":
            out += String("\\\\")
        elif ch == "\"":
            out += String("\\\"")
        elif ch == "\n":
            out += String("\\n")
        elif ch == "\r":
            out += String("\\r")
        elif ch == "\t":
            out += String("\\t")
        else:
            out += String(ch)
        i += 1
    return out

fn json_string(s: String) -> String:
    return String("\"") + _escape_json_str(s) + String("\"")

fn json_array_str(items: List[String]) -> String:
    var out = String("[")
    var i = 0
    while i < len(items):
        if i > 0: out += String(",")
        out += items[i]
        i += 1
    out += String("]"); return out

fn json_number_array(xs: List[Float64]) -> String:
    var parts = List[String](); var i = 0
    while i < len(xs): parts.append(String(xs[i])); i += 1
    return json_array_str(parts)

fn json_string_array(xs: List[String]) -> String:
    var parts = List[String](); var i = 0
    while i < len(xs): parts.append(json_string(xs[i])); i += 1
    return json_array_str(parts)

 

# src/momijo/visual/core/core.mojo

fn _escape_html(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "&":  out += String("&amp;")
        elif ch == "<": out += String("&lt;")
        elif ch == ">": out += String("&gt;")
        elif ch == "\"": out += String("&quot;")
        else: out += String(ch)
        i += 1
    return out

# ... keep your _escape_html as-is ...

fn echarts_html(
    option_json: String,
    title: String = "ECharts Demo",
    width_px: Int = 960,
    height_px: Int = 750,
    include_world_map: Bool = False,   # kept for compatibility; unused
    full_page: Bool = True,
    include_gl: Bool = False,          # load echarts-gl for 3D
    pre_init_js: String = "",          # JS to run after echarts loads, before init/setOption
    include_wordcloud_plugin: Bool = False   # NEW: load echarts-wordcloud after echarts
) -> String:
    var html = String("")
    html += String("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n")
    html += String("<meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
    html += String("<title>") + _escape_html(title) + String("</title>\n<style>\n")
    if full_page:
        html += String("html,body{height:100%;width:100%;margin:0;padding:0;background:#f5f5f5;font-family:system-ui,sans-serif}\n#main{position:fixed;inset:0;}\n")
    else:
        html += String("html,body{margin:0;padding:0;background:#f5f5f5;font-family:system-ui,sans-serif}\n#main{width:") + String(width_px) + String("px;height:") + String(height_px) + String("px;margin:0 auto;}\n")
    html += String("#fallback{display:none;padding:16px;color:#222}\n#fallback .hint{font-size:14px;opacity:.8}\n</style>\n</head>\n<body>\n")
    html += String("<div id=\"main\"></div>\n")
    html += String("<div id=\"fallback\"><h3>Could not load ECharts</h3><div class=\"hint\">")
    html += String("Load failed from local files & CDNs. Put <code>echarts.min.js</code>")
    html += String(" (and <code>echarts-gl.min.js</code> for 3D, <code>echarts-wordcloud.min.js</code> for WordCloud) next to this HTML, or allow network access.</div></div>\n")

    # build loader
    html += String("<script>(function(){\n")
    html += String("  const local=['echarts.min.js'];\n")
    html += String("  const localGL=['echarts-gl.min.js'];\n")
    html += String("  const localWC=['echarts-wordcloud.min.js'];\n")
    html += String("  const cdns=[\n")
    html += String("    'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js',\n")
    html += String("    'https://unpkg.com/echarts@5/dist/echarts.min.js',\n")
    html += String("    'https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js'\n")
    html += String("  ];\n")
    html += String("  const cdnsGL=[\n")
    html += String("    'https://cdn.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js',\n")
    html += String("    'https://unpkg.com/echarts-gl@2/dist/echarts-gl.min.js',\n")
    html += String("    'https://cdnjs.cloudflare.com/ajax/libs/echarts-gl/2.0.9/echarts-gl.min.js'\n")
    html += String("  ];\n")
    html += String("  const cdnsWC=[\n")
    html += String("    'https://cdn.jsdelivr.net/npm/echarts-wordcloud@2/dist/echarts-wordcloud.min.js',\n")
    html += String("    'https://unpkg.com/echarts-wordcloud@2/dist/echarts-wordcloud.min.js',\n")
    html += String("    'https://cdnjs.cloudflare.com/ajax/libs/echarts-wordcloud/2.0.0/echarts-wordcloud.min.js'\n")
    html += String("  ];\n")

    html += String("  function showFallback(){ document.getElementById('fallback').style.display='block'; }\n")
    html += String("  let myChart=null;\n")
    html += String("  function initChart(){\n")
    html += String("    if(!window.echarts){ showFallback(); return; }\n")
    if len(pre_init_js) > 0:
        html += String("    try{ ") + pre_init_js + String(" }catch(__e){ console.warn('pre_init_js error:', __e); }\n")
    html += String("    const el=document.getElementById('main'); myChart=echarts.init(el);\n")
    html += String("    var option=") + option_json + String(";\n")
    html += String("    try{ myChart.setOption(option); }catch(e){ console.error(e); showFallback(); }\n")
    html += String("    window.addEventListener('resize', function(){ if(myChart){ myChart.resize(); } });\n")
    html += String("  }\n")

    html += String("  function loadScript(url, ok, fail){ var s=document.createElement('script'); s.src=url; s.onload=ok; s.onerror=function(){ s.remove(); fail&&fail(); }; document.head.appendChild(s); }\n")
    html += String("  function chain(list, next){ let i=0; (function step(){ if(i>=list.length){ next(false); return; } loadScript(list[i++], function(){ next(true); }, step); })(); }\n")

    # sequence: echarts -> (optional gl) -> (optional wordcloud) -> init
    html += String("  function afterEcharts(){\n")
    html += String("    function afterGL(){\n")
    html += String("      function afterWC(){ initChart(); }\n")
    html += String("      ") + (String("chain(localWC, function(haveL){ if(haveL){ afterWC(); } else { chain(cdnsWC, function(haveC){ if(haveC){ afterWC(); } else { showFallback(); } }); } });") if include_wordcloud_plugin else String("afterWC();")) + String("\n")
    html += String("    }\n")
    html += String("    ") + (String("chain(localGL, function(haveL){ if(haveL){ afterGL(); } else { chain(cdnsGL, function(haveC){ if(haveC){ afterGL(); } else { showFallback(); } }); } });") if include_gl else String("afterGL();")) + String("\n")
    html += String("  }\n")

    html += String("  chain(local, function(haveLocal){ if(haveLocal){ afterEcharts(); } else { chain(cdns, function(haveCdn){ if(haveCdn){ afterEcharts(); } else { showFallback(); } }); } });\n")
    html += String("})();</script>\n")
    html += String("</body>\n</html>\n")
    return html



fn write_text(path: String, text: String) -> Bool:
    try:
        var f = open(path, "wb")
        var i = 0
        while i < len(text):
            var b = UInt8(text[i])
            f.write(b)
            i += 1
        f.close(); return True
    except _:
        return False