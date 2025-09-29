# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/html_viewer.mojo
# Description: ECharts HTML renderer for Mojo (SVG/canvas; CDN fallback; downloads; JSON-safe).

from python import Python

# ------------------------------ IO helpers ------------------------------

fn _write_text_utf8(path: String, data: String) raises:
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "w", encoding="utf-8")
    f.write(data)
    f.close()

fn _abs_file_url(path: String) raises -> String:
    var os = Python.import_module("os")
    var full = os.path.abspath(path)
    return "file://" + String(full)

# ------------------------------ Public API ------------------------------

fn render_echarts(
    opts_json: String,
    outfile: String,
    title: String = "Chart",
    div_id: String = "chart",
    width: String = "100%",
    height: String = "480px",
    renderer: String = "svg",      # svg is safer than canvas in many VMs
    theme: String = "",            # empty => null
    enable_download: Bool = True,
    auto_open: Bool = True
) raises:
    var html = String("")

    # ---------- HEAD ----------
    html += "<!doctype html><html><head><meta charset=\"utf-8\">"
    html += "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    html += "<title>" + title + "</title>"
    html += "<style>"
    html += "body{margin:0;font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif}"
    html += ".toolbar{display:flex;gap:.5rem;align-items:center;padding:.5rem;border-bottom:1px solid #eee}"
    html += ".btn{padding:.35rem .6rem;border:1px solid #ddd;border-radius:.5rem;cursor:pointer;background:#fafafa}"
    html += ".btn:hover{background:#f0f0f0}"
    html += "</style>"
    html += "<script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\" "
    html += "onerror=\"var s=document.createElement('script');"
    html += "s.src='https://unpkg.com/echarts@5/dist/echarts.min.js';document.head.appendChild(s);\"></script>"
    html += "</head><body>"

    # ---------- ERROR BOX ----------
    html += "<div id=\"_echarts_error\" style=\"display:none;padding:.75rem;margin:.5rem;"
    html += "border:1px solid #f99;background:#fff4f4;color:#900;border-radius:.5rem;font-family:monospace\"></div>"

    # ---------- TOOLBAR ----------
    if enable_download:
        html += "<div class=\"toolbar\">"
        html += "<button class=\"btn\" onclick=\"downloadJSON()\">Download JSON</button>"
        html += "<button class=\"btn\" onclick=\"downloadImage('png')\">Download PNG</button>"
        html += "<button class=\"btn\" onclick=\"downloadImage('svg')\">Download SVG</button>"
        html += "</div>"

    # ---------- CHART CONTAINER ----------
    html += "<div id=\"" + div_id + "\" style=\"width:" + width + ";height:" + height + ";\"></div>"

    # ---------- OPTIONS JSON (safe container) ----------
    html += "<script id=\"echarts_opts\" type=\"application/json\">" + opts_json + "</script>"

    # ---------- THEME STRING ----------
    var theme_js = String("null")
    if len(theme) > 0:
        theme_js = "'" + theme + "'"

    # ---------- SCRIPT ----------
    
html += "<script>
(async function(){
  function showErr(msg){ try{ console.error(msg); }catch(_){ } }
  // Wait for world map if geo is used
  try {
    var needWorld = (typeof opts==='string' ? opts.indexOf("'map':'world'")>=0 || opts.indexOf('"map":"world"')>=0 : false);
    if(needWorld && typeof echarts!=='undefined'){
      let sources = [
        'https://fastly.jsdelivr.net/gh/apache/echarts-website@asf-site/examples/data/asset/geo/world.json',
        'https://fastly.jsdelivr.net/npm/echarts-maps@1.0.0/world.json',
        'https://cdn.jsdelivr.net/npm/echarts-maps@1.0.0/world.json'
      ];
      let ok = false;
      for(const u of sources){
        try{
          const r = await fetch(u);
          if(r.ok){
            const gj = await r.json();
            echarts.registerMap('world', gj);
            ok = true; break;
          }
        }catch(e){}
      }
      if(!ok){ showErr('Failed to load world geojson'); }
    }
  } catch(e){ showErr(e); }

  try {
    var el = document.getElementById('chart');
    var chart = echarts.init(el, null, {renderer:'canvas'});
    var opts = window.__MOMIJO_OPTION; // string JSON
    if(!opts){ showErr('Missing option'); return; }
    var obj;
    try{ obj = eval('(' + opts + ')'); }catch(_){ obj = JSON.parse(opts.replace(/'/g,'"')); }
    chart.setOption(obj, true);
    window.addEventListener('resize', function(){ try{ chart.resize(); }catch(_){ } });
  } catch(e){ showErr(e&&e.stack?e.stack:String(e)); }
})();
</script>
"

    html += "</body></html>"

    _write_text_utf8(outfile, html)

    if auto_open:
        var wb = Python.import_module("webbrowser")
        wb.open(_abs_file_url(outfile))