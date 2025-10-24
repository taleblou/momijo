# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/html/preview.mojo

from python import Python
from momijo.visual.core.core import echarts_html

fn open_in_browser(file_path: String):
    try:
        var os = Python.import_module("os")
        var abspath = String(os.path.abspath(file_path))
        var url = String("file://") + abspath
        var wb = Python.import_module("webbrowser")
        wb.open(url)
    except _:
        pass

fn _py_write_abs(path: String, content: String) -> Bool:
    try:
        var io = Python.import_module("io")
        var f = io.open(path, "w", encoding="utf-8")
        f.write(content)
        f.close()
        return True
    except _:
        return False

fn write_preview_html(html: String, save: Bool, out_path: String) -> String:
    try:
        var os = Python.import_module("os")
        if save:
            var abs_path = String(os.path.abspath(out_path))
            var ok = _py_write_abs(abs_path, html)
            if ok: return abs_path
        var tmp_base = os.getenv("TMPDIR")
        var base = tmp_base if tmp_base else "/tmp"
        var path = String(base) + String("/momijo_preview.html")
        _ = _py_write_abs(path, html)
        return path
    except _:
        return String("preview.html")

fn render_html(option_json: String,
               title: String = String("momijo • ECharts"),
               width_px: Int = 960,
               height_px: Int = 540,
               include_world_map: Bool = False,
               save: Bool = True,
               auto_open: Bool = True,
               out_path: String = String("index.html"),
               full_page: Bool = True,
               include_gl: Bool = False,
               pre_init_js: String = String(""),
               include_wordcloud_plugin: Bool = False) -> String:   # NEW
    var html = echarts_html(option_json,
                            title,
                            width_px,
                            height_px,
                            include_world_map,
                            full_page,
                            include_gl,
                            pre_init_js,
                            include_wordcloud_plugin)  # pass through
    var path = write_preview_html(html, save, out_path)
    if auto_open:
        open_in_browser(path)
    return path