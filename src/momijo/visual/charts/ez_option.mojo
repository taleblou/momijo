# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/charts/ez_option.mojo
# Description: Ergonomic DSL wrapper around Option.

from momijo.visual.charts.option import Option

# ---- helpers for contour lines (file-scope, not nested) ----------------------

fn _lerp(a: Float64, b: Float64, ta: Float64, tb: Float64, t: Float64) -> Float64:
    var denom = tb - ta
    if denom == 0.0:
        return a
    var u = (t - ta) / denom
    return a + u * (b - a)

# edge: 0=bottom, 1=right, 2=top, 3=left
fn _edge_point(edge: Int,
               xL: Float64, xR: Float64, yB: Float64, yT: Float64,
               z00: Float64, z10: Float64, z01: Float64, z11: Float64,
               lvl: Float64) -> (Float64, Float64):
    if edge == 0:
        return ( _lerp(xL, xR, z00, z10, lvl), yB )
    elif edge == 1:
        return ( xR, _lerp(yB, yT, z10, z11, lvl) )
    elif edge == 2:
        return ( _lerp(xL, xR, z01, z11, lvl), yT )
    else:
        return ( xL, _lerp(yB, yT, z00, z01, lvl) )


fn _escape_str(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "\\": out += String("\\\\")
        elif ch == "\"": out += String("\\\"")
        elif ch == "\n": out += String("\\n")
        elif ch == "\r": out += String("\\r")
        elif ch == "\t": out += String("\\t")
        else: out += String(ch)
        i += 1
    return out

fn _json_str(s: String) -> String:
    return String("'") + _escape_str(s) + String("'")

fn _json_array_str(xs: List[String]) -> String:
    var out = String("["); var i = 0
    while i < len(xs):
        out += _json_str(xs[i])
        if i + 1 < len(xs): out += String(",")
        i += 1
    out += String("]")
    return out

fn _json_array_int(xs: List[Int]) -> String:
    var out = String("["); var i = 0
    while i < len(xs):
        out += String(xs[i])
        if i + 1 < len(xs): out += String(",")
        i += 1
    out += String("]")
    return out
fn _append_segment(mut s: String,
                                    mut is_first: Bool,
                                    p1x: Float64, p1y: Float64,
                                    p2x: Float64, p2y: Float64):
        if not is_first:
            s += String(",")
        s += String("[") + String(p1x) + String(",") + String(p1y) + String("],")
        s += String("[") + String(p2x) + String(",") + String(p2y) + String("], null")
        is_first = False

# Add this tiny helper near the top of the file (outside the struct), or inside if you prefer:
fn _format_scaled_6(scaled: Int) -> String:
    # scaled represents a fixed-point value with 6 decimals, i.e., value * 1_000_000
    var one_million = 1000000
    var intp = scaled // one_million
    var frac = scaled - intp * one_million
    var frac_str = String(frac)
    # left-pad to exactly 6 digits
    var pad = 6 - len(frac_str)
    var buf = String("")
    var i = 0
    while i < pad:
        buf += String("0")
        i += 1
    buf += frac_str
    return String(intp) + String(".") + buf

struct EasyOption:
    var inner: Option
    var _titles: List[String]
    var _extra_kv: List[String]
    var _x3d: String
    var _y3d: String
    var _z3d: String
    var _grid3d: String 
    var _parallel_axes: List[String]
    var _parallel_obj: String

    fn __init__(out self):
        self.inner = Option()
        self._titles = List[String]()
        self._extra_kv = List[String]()  
        self._x3d = String("")
        self._y3d = String("")
        self._z3d = String("")
        self._grid3d = String("") 
        self._parallel_axes = List[String]()
        self._parallel_obj = String("")

    fn set_color(mut self, colors: List[String]):
        self.inner.set_colors(colors)

    fn set_background(mut self, css_color: String):
        # Pass raw color; Option handles quoting/JSON appropriately.
        self.inner.set_background(css_color)

    fn add_title(mut self, text: String, left: String = "center", top: Int = 10,
                 textStyle_fontSize: Int = 16, textStyle_fontWeight: String = "normal"):
        var obj = String("{ 'text':") + _json_str(text) +
                  String(", 'left':") + _json_str(left) +
                  String(", 'top':") + String(top) +
                  String(", 'textStyle':{ 'fontSize':") + String(textStyle_fontSize) +
                  String(", 'fontWeight':") + _json_str(textStyle_fontWeight) +
                  String(" } }")
        self._titles.append(obj)

    fn set_title(mut self, text: String, left: String = "center", top: Int = 10,
                 textStyle_fontSize: Int = 16, textStyle_fontWeight: String = "normal"):
        self.add_title(text, left, top, textStyle_fontSize, textStyle_fontWeight)

    fn set_tooltip_axis_shadow(mut self, bg: String = "#fff", border: String = "#aaa",
                               border_width: Int = 1, text_color: String = "#000",
                               formatter_js: String = ""):
        var tip = String("{ 'trigger':'axis', 'axisPointer':{ 'type':'shadow' }, ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)

    fn set_legend(mut self, names: List[String], top_px: Int = 80, font_size: Int = 12, color: String = "#333"):
        var obj = String("{ 'data':") + _json_array_str(names) +
                  String(", 'top':") + String(top_px) +
                  String(", 'textStyle':{ 'fontSize':") + String(font_size) +
                  String(", 'color':") + _json_str(color) + String(" } }")
        self.inner.set_legend(obj)

    fn set_toolbox_default(mut self, right_px: Int = 20, top_px: Int = 80, show: Bool = True):
        var obj = String("{ 'show':") + (String("true") if show else String("false")) +
                  String(", 'feature':{ 'saveAsImage':{ 'title':'Save as Image' }, 'dataView':{ 'readOnly':false }, ") +
                  String("'magicType':{ 'type':['line','bar','stack'] }, 'restore':{}, 'dataZoom':{} }, ") +
                  String("'right':") + String(right_px) + String(", 'top':") + String(top_px) + String(" }")
        self.inner.set_toolbox(obj)

    fn set_grid(mut self, top_px: Int, left_pct: String, right_pct: String, bottom_pct: String, contain_label: Bool = True):
        var obj = String("{ 'top':") + String(top_px) +
                  String(", 'left':") + _json_str(left_pct) +
                  String(", 'right':") + _json_str(right_pct) +
                  String(", 'bottom':") + _json_str(bottom_pct) +
                  String(", 'containLabel':") + (String("true") if contain_label else String("false")) +
                  String(" }")
        self.inner.set_grid(obj)

    fn set_xaxis_category_json(mut self, name: String, name_location: String, name_gap: Int,
                               labels_color: String, rotate_deg: Int, data: List[String]):
        var obj = String("{ 'type':'category', 'name':") + _json_str(name) +
                  String(", 'nameLocation':") + _json_str(name_location) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLabel':{ 'color':") + _json_str(labels_color) +
                  String(", 'rotate':") + String(rotate_deg) + String(" }, ") +
                  String(" 'data':") + _json_array_str(data) + String(" }")
        self.inner.set_xaxis_json(obj)

    fn set_yaxis_value_json(mut self, name: String, name_location: String, name_gap: Int, labels_color: String):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameLocation':") + _json_str(name_location) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLabel':{ 'color':") + _json_str(labels_color) + String(" } }")
        self.inner.set_yaxis_json(obj)

    fn add_datazoom_slider(mut self, start_pct: Int = 0, end_pct: Int = 100):
        self.inner.add_datazoom_json(String("{ 'type':'slider','start':") + String(start_pct) +
                                     String(", 'end':") + String(end_pct) + String(" }"))

    fn add_datazoom_inside(mut self):
        self.inner.add_datazoom_json(String("{ 'type':'inside' }"))

    fn set_visualmap(mut self, minv: Int, maxv: Int, dimension: Int, inrange_colors: List[String]):
        var obj = String("{ 'show':false, 'min':") + String(minv) +
                  String(", 'max':") + String(maxv) +
                  String(", 'dimension':") + String(dimension) +
                  String(", 'inRange':{ 'color':") + _json_array_str(inrange_colors) + String(" } }")
        self.inner.set_visual_map(obj)

    fn add_bar_series(mut self, name: String, data: List[Int], color: String,
                      show_labels: Bool = True, label_fmt: String = "{c} USD",
                      add_mark_point_minmax: Bool = True, add_mark_line_avg: Bool = True):
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'bar', 'data':") + _json_array_int(data)
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        if add_mark_point_minmax:
            obj += String(", 'markPoint':{ 'data':[ { 'type':'max','name':'Max' }, { 'type':'min','name':'Min' } ] }")
        if add_mark_line_avg:
            obj += String(", 'markLine':{ 'data':[ { 'type':'average','name':'Avg' } ] }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" } }")
        self.inner.add_series_json(obj)

    fn set_animation(mut self, enabled: Bool, duration_ms: Int, easing: String, delay_per_index_ms: Int = 50):
        self.inner.set_animation(enabled, duration_ms, String(easing))
        # Only inject animationDelay to avoid duplicating animationEasing
        self._extra_kv.append(String("'animationDelay':function(idx){ return idx * ") + String(delay_per_index_ms) + String("; }"))

    fn add_extra_kv(mut self, kv_fragment: String):
        self._extra_kv.append(kv_fragment)

    fn to_json(mut self) -> String:
        if len(self._titles) > 0:
            var t = String("["); var i = 0
            while i < len(self._titles):
                t += self._titles[i]
                if i + 1 < len(self._titles): t += String(",")
                i += 1
            t += String("]")
            self.inner.set_title(t)

        var j = self.inner.to_json()
        if len(self._extra_kv) > 0 and len(j) > 1 and j[0] == "{" and j[len(j)-1] == "}":
            var tail = String(""); var k = 0
            while k < len(self._extra_kv):
                tail += String(", ") + self._extra_kv[k]
                k += 1
            j = j[0:len(j)-1] + tail + String("}")

                # Inject parallel & parallelAxis if present
        if len(self._parallel_obj) > 0:
            self._extra_kv.append(String("'parallel': ") + self._parallel_obj)
        if len(self._parallel_axes) > 0:
            # build [ ... ] from _parallel_axes
            var pa = String("[")
            var i3 = 0
            while i3 < len(self._parallel_axes):
                pa += self._parallel_axes[i3]
                if i3 + 1 < len(self._parallel_axes): pa += String(",")
                i3 += 1
            pa += String("]")
            self._extra_kv.append(String("'parallelAxis': ") + pa)
        return j


    # Line series
    fn add_line_series(mut self,
                       name: String,
                       data: List[Int],
                       color: String,
                       smooth: Bool = False,
                       show_symbol: Bool = True,
                       line_width: Int = 2,
                       area_opacity: Int = 0,     # 0..100 (0 = no area)
                       show_labels: Bool = False,
                       label_fmt: String = "{c}"):

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'line', 'data':") + _json_array_int(data)

        if smooth:
            obj += String(", 'smooth':true")

        if show_symbol == False:
            obj += String(", 'showSymbol':false")

        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")

        if area_opacity > 0:
            # convert 0..100 to 0..1 in JS
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")

        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")

        obj += String(" }")
        self.inner.add_series_json(obj)


    fn set_tooltip_axis_line(mut self,
                                bg: String = "#fff",
                                border: String = "#aaa",
                                border_width: Int = 1,
                                text_color: String = "#000",
                                formatter_js: String = ""):
        var tip = String("{ 'trigger':'axis', 'axisPointer':{ 'type':'line' }, ") +
                    String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                    String(" 'borderColor':") + _json_str(border) + String(",") +
                    String(" 'borderWidth':") + String(border_width) + String(",") +
                    String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)

    # Scatter series (value-value axes). Builds data = [[x0,y0],[x1,y1],...].
    fn add_scatter_series(mut self,
                          name: String,
                          xs: List[Int],
                          ys: List[Int],
                          color: String,
                          symbol_size: Int = 8,
                          opacity: Int = 100,         # 0..100
                          show_labels: Bool = False,
                          label_fmt: String = "({c})"):
        # Build [[x,y], ...] manually.
        var data_json = String("[")
        var n = len(xs)
        var i = 0
        while i < n and i < len(ys):
            data_json += String("[") + String(xs[i]) + String(",") + String(ys[i]) + String("]")
            if i + 1 < n and i + 1 < len(ys): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'scatter', 'data':") + data_json
        obj += String(", 'symbolSize':") + String(symbol_size)
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(", 'opacity':") + String(opacity) + String(" / 100 }")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    fn set_tooltip_axis_cross(mut self,
                              bg: String = "#fff",
                              border: String = "#aaa",
                              border_width: Int = 1,
                              text_color: String = "#000",
                              formatter_js: String = ""):
        var tip = String("{ 'trigger':'axis', 'axisPointer':{ 'type':'cross' }, ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)


    # Tooltip for item-based charts (pie, funnel, etc.)
    fn set_tooltip_item(mut self,
                        bg: String = "#fff",
                        border: String = "#aaa",
                        border_width: Int = 1,
                        text_color: String = "#000",
                        formatter_js: String = ""):
        var tip = String("{ 'trigger':'item', ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)

    # Pie / Donut series
    # - labels: category names
    # - values: numeric values (same length as labels)
    # - radius_outer_pct: e.g. 70  -> '70%'
    # - radius_inner_pct: e.g. 0   -> '0%' (donut if > 0)
    # - rose_type: 'radius' or 'area' or '' (normal pie)
    fn add_pie_series(mut self,
                      name: String,
                      labels: List[String],
                      values: List[Int],
                      radius_outer_pct: Int = 70,
                      radius_inner_pct: Int = 0,
                      center_x: String = "50%",
                      center_y: String = "55%",
                      rose_type: String = "",
                      label_show: Bool = True,
                      label_formatter: String = "{b}: {c} ({d}%)",
                      item_opacity: Int = 100):
        # data = [{name:'A', value:10}, ...]
        var data_json = String("[")
        var n = len(labels)
        var i = 0
        while i < n and i < len(values):
            data_json += String("{ 'name':") + _json_str(labels[i]) +
                         String(", 'value':") + String(values[i]) + String(" }")
            if i + 1 < n and i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'pie'")
        # radius can be a single value or array [inner, outer]
        obj += String(", 'radius':[ '") + String(radius_inner_pct) + String("%','") +
               String(radius_outer_pct) + String("%' ]")
        obj += String(", 'center':[ ") + _json_str(center_x) + String(", ") + _json_str(center_y) + String(" ]")
        obj += String(", 'data':") + data_json
        if len(rose_type) > 0:
            obj += String(", 'roseType':") + _json_str(rose_type)
        if label_show:
            obj += String(", 'label':{ 'show':true, 'formatter':") + _json_str(label_formatter) + String(" }")
        obj += String(", 'itemStyle':{ 'opacity':") + String(item_opacity) + String(" / 100 }")
        obj += String(" }")

        self.inner.add_series_json(obj)
        
    # Heatmap series from triplets (x_idx, y_idx, value).
    # x_idx, y_idx, values must have the same length.
    fn add_heatmap_series_from_triplets(
        mut self,
        name: String,
        x_idx: List[Int],
        y_idx: List[Int],
        values: List[Int],
        label_show: Bool = False,
        label_fmt: String = "{c}",
        border_width: Int = 1
    ):
        var n = len(x_idx)
        if len(y_idx) < n: n = len(y_idx)
        if len(values) < n: n = len(values)

        # Build data: [ [xi, yi, v], ... ]
        var data_json = String("[")
        var i = 0
        while i < n:
            data_json += String("[") + String(x_idx[i]) + String(",") + String(y_idx[i]) + String(",") + String(values[i]) + String("]")
            if i + 1 < n: data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'heatmap', 'data':") + data_json
        if label_show:
            obj += String(", 'label':{ 'show':true, 'formatter':") + _json_str(label_fmt) + String(" }")
        if border_width > 0:
            obj += String(", 'itemStyle':{ 'borderWidth':") + String(border_width) + String(" }")
        # Emphasis for better focus on hover
        obj += String(", 'emphasis':{ 'itemStyle':{ 'shadowBlur':10, 'shadowColor':'rgba(0,0,0,0.4)' } }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    # Convenience visualMap for heatmap (continuous gradient on value dimension = 2)
    # colors: gradient stops from low -> high
    fn set_visualmap_continuous(mut self, minv: Int, maxv: Int, colors: List[String], dimension: Int = 2):
        var obj = String("{ 'min':") + String(minv) +
                  String(", 'max':") + String(maxv) +
                  String(", 'calculable':true, 'realtime':true, 'orient':'vertical', 'left':'right', 'top':'center'") +
                  String(", 'inRange':{ 'color':") + _json_array_str(colors) + String(" }, 'dimension':") + String(dimension) + String(" }")
        self.inner.set_visual_map(obj)


    # Boxplot series using five-number summaries per category.
    # labels: x-axis categories (same length as fivenum)
    # fivenum: [[min,Q1,median,Q3,max], ...]  (length == len(labels))
    # box_width_px: [min,max] pixel width range for boxes (ECharts accepts numbers/percents)
    fn add_boxplot_series_from_fivenum(
        mut self,
        name: String,
        labels: List[String],
        fivenum: List[List[Int]],
        color: String = "#5470C6",
        box_border_width: Int = 1,
        whisker_color: String = "#333",
        show_labels: Bool = False,
        label_fmt: String = "{c}",
        box_width_px_min: Int = 7,
        box_width_px_max: Int = 50
    ):
        # Build data: [ [min,Q1,median,Q3,max], ... ]
        var data_json = String("[")
        var i = 0
        while i < len(fivenum) and i < len(labels):
            # serialize one five-number vector
            var item = fivenum[i]
            if len(item) >= 5:
                data_json += String("[") + String(item[0]) + String(",") +
                             String(item[1]) + String(",") +
                             String(item[2]) + String(",") +
                             String(item[3]) + String(",") +
                             String(item[4]) + String("]")
                if i + 1 < len(fivenum) and i + 1 < len(labels): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'boxplot'")
        obj += String(", 'data':") + data_json
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) +
               String(", 'borderColor':") + _json_str(whisker_color) +
               String(", 'borderWidth':") + String(box_border_width) + String(" }")
        obj += String(", 'boxWidth':[ ") + String(box_width_px_min) + String(", ") + String(box_width_px_max) + String(" ]")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")

        # Ensure xAxis is category with these labels (demo usually sets it explicitly, but this helps)
        # NOTE: If the demo already set x-axis, this won't override it.
        var lbl_json = String("[")
        var j = 0
        while j < len(labels):
            lbl_json += _json_str(labels[j])
            if j + 1 < len(labels): lbl_json += String(",")
            j += 1
        lbl_json += String("]")

        # Only set x-axis categories if not already set (caller can still override explicitly after)
        self.inner.set_xaxis_json(String("{ 'type':'category', 'name':'', 'axisLabel':{ 'color':'#333' }, 'data':") + lbl_json + String(" }"))

        # y-axis numeric (caller can override)
        self.inner.set_yaxis_json(String("{ 'type':'value', 'name':'', 'axisLabel':{ 'color':'#333' } }"))

        self.inner.add_series_json(obj)

    # Optional scatter series for boxplot outliers.
    # xs: category index per outlier (0-based), ys: value per outlier
    fn add_boxplot_outliers(
        mut self,
        name: String,
        xs: List[Int],
        ys: List[Int],
        color: String = "#EE6666",
        symbol_size: Int = 8,
        opacity: Int = 85
    ):
        var n = len(xs)
        if len(ys) < n: n = len(ys)

        # Build data as [[xIdx, y], ...]
        var data_json = String("[")
        var i = 0
        while i < n:
            data_json += String("[") + String(xs[i]) + String(",") + String(ys[i]) + String("]")
            if i + 1 < n: data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'scatter', 'data':") + data_json
        obj += String(", 'symbolSize':") + String(symbol_size)
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(", 'opacity':") + String(opacity) + String(" / 100 }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    # Handy tooltip for boxplot (trigger item).
    fn set_tooltip_boxplot_item(
        mut self,
        bg: String = "#fff",
        border: String = "#aaa",
        border_width: Int = 1,
        text_color: String = "#000"
    ):
        # p.data = [min,Q1,median,Q3,max]
        var f = String(
            "function(p){ var d=p.data; " +
            "return '<b>'+p.name+'</b><br/>'+" +
            "'min: '+d[0]+'<br/>'+" +
            "'Q1: '+d[1]+'<br/>'+" +
            "'median: '+d[2]+'<br/>'+" +
            "'Q3: '+d[3]+'<br/>'+" +
            "'max: '+d[4]; }"
        )
        var tip = String("{ 'trigger':'item', ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }, ") +
                  String(" 'formatter': ") + f + String(" }")
        self.inner.set_tooltip(tip)


        # Stem (lollipop) plot using ECharts 'custom' series.
    # xcats: categories for x-axis
    # y:    values per category
    # baseline: vertical stems start from this y (e.g. 0)
    fn add_stem_series(
        mut self,
        name: String,
        xcats: List[String],
        y: List[Int],
        color: String = "#5470C6",
        baseline: Int = 0,
        line_width: Int = 2,
        marker_radius: Int = 5
    ):
        # Ensure x-axis categories are set
        var lbl = String("[")
        var i = 0
        while i < len(xcats):
            lbl += _json_str(xcats[i])
            if i + 1 < len(xcats): lbl += String(",")
            i += 1
        lbl += String("]")
        self.inner.set_xaxis_json(
            String("{ 'type':'category', 'name':'', 'axisLabel':{ 'color':'#333' }, 'data':") + lbl + String(" }")
        )

        # y-axis numeric (caller can still override after)
        self.inner.set_yaxis_json(String("{ 'type':'value', 'name':'', 'axisLabel':{ 'color':'#333' } }"))

        # Build data: [ [xIndex, y], ... ]
        var data = String("[")
        var n = len(xcats); 
        if len(y) < n: 
            n = len(y)
        var k = 0
        while k < n:
            data += String("[") + String(k) + String(",") + String(y[k]) + String("]")
            if k + 1 < n: data += String(",")
            k += 1
        data += String("]")

        # JS renderItem draws the stem line + cap circle at (x, y)
        var render = String(
            "function(params, api){" +
            "  var x = api.value(0), y = api.value(1);" +
            "  var pTop = api.coord([x, y]);" +
            "  var pBase = api.coord([x, " + String(baseline) + "]);" +
            "  return {" +
            "    type:'group'," +
            "    children:[" +
            "      {type:'line', shape:{x1:pBase[0], y1:pBase[1], x2:pTop[0], y2:pTop[1]}," +
            "       style:{stroke:'" + color + "', lineWidth:" + String(line_width) + "}}," +
            "      {type:'circle', shape:{cx:pTop[0], cy:pTop[1], r:" + String(marker_radius) + "}," +
            "       style:{fill:'" + color + "'}}" +
            "    ]};" +
            "}"
        )

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'custom'")
        obj += String(", 'data':") + data
        obj += String(", 'encode':{ 'x':0, 'y':1 }")
        obj += String(", 'renderItem': ") + render
        obj += String(", 'tooltip':{ 'trigger':'item' }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    # Errorbar (vertical) using ECharts 'custom' series.
    # xcats: category labels along x
    # y:     central value per category (for tooltip reference; not drawn here)
    # low:   lower bound per category
    # high:  upper bound per category
    # Note: data length must match across arrays; x index is 0-based.
    fn add_errorbar_series(
        mut self,
        name: String,
        xcats: List[String],
        y: List[Int],
        low: List[Int],
        high: List[Int],
        color: String = "#333",
        line_width: Int = 2,
        cap_width_px: Int = 12
    ):
        # Ensure x-axis categories exist (caller can override later if needed)
        var lbl = String("[")
        var i = 0
        while i < len(xcats):
            lbl += _json_str(xcats[i])
            if i + 1 < len(xcats): lbl += String(",")
            i += 1
        lbl += String("]")
        self.inner.set_xaxis_json(
            String("{ 'type':'category', 'name':'', 'axisLabel':{ 'color':'#333' }, 'data':") + lbl + String(" }")
        )
        self.inner.set_yaxis_json(String("{ 'type':'value', 'name':'', 'axisLabel':{ 'color':'#333' } }"))

        # Build data as [xIdx, y, low, high]
        var n = len(xcats)
        if len(y)    < n: n = len(y)
        if len(low)  < n: n = len(low)
        if len(high) < n: n = len(high)

        var data = String("[")
        var k = 0
        while k < n:
            data += String("[") + String(k) + String(",") + String(y[k]) + String(",") + String(low[k]) + String(",") + String(high[k]) + String("]")
            if k + 1 < n: data += String(",")
            k += 1
        data += String("]")

        # renderItem draws: vertical line (low->high) + two caps at low/high
        var render = String(
            "function(params, api){" +
            "  var x = api.value(0), y = api.value(1), lo = api.value(2), hi = api.value(3);" +
            "  var pLo = api.coord([x, lo]);" +
            "  var pHi = api.coord([x, hi]);" +
            "  var half = " + String(cap_width_px) + String(" / 2;" ) +
            "  var line = {type:'line', shape:{x1:pLo[0], y1:pLo[1], x2:pHi[0], y2:pHi[1]}, style:{stroke:'" + color + "', lineWidth:" + String(line_width) + "}};" +
            "  var capLo = {type:'line', shape:{x1:pLo[0]-half, y1:pLo[1], x2:pLo[0]+half, y2:pLo[1]}, style:{stroke:'" + color + "', lineWidth:" + String(line_width) + "}};" +
            "  var capHi = {type:'line', shape:{x1:pHi[0]-half, y1:pHi[1], x2:pHi[0]+half, y2:pHi[1]}, style:{stroke:'" + color + "', lineWidth:" + String(line_width) + "}};" +
            "  return {type:'group', children:[line, capLo, capHi]};" +
            "}"
        )

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'custom'")
        obj += String(", 'data':") + data
        obj += String(", 'encode':{ 'x':0, 'y':1 }")
        obj += String(", 'renderItem': ") + render
        obj += String(", 'tooltip':{ 'trigger':'item', 'formatter':") +
               String("function(p){var v=p.value;return '<b>'+p.name+'</b><br/>'+'y: '+v[1]+'<br/>low: '+v[2]+'<br/>high: '+v[3];}") +
               String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    # Convenience: symmetric errors: y ± err
    fn add_errorbar_series_symmetric(
        mut self,
        name: String,
        xcats: List[String],
        y: List[Int],
        err: List[Int],
        color: String = "#333",
        line_width: Int = 2,
        cap_width_px: Int = 12
    ):
        var low = List[Int]()
        var high = List[Int]()
        var n = len(y)
        if len(err) < n: n = len(err)
        var i2 = 0
        while i2 < n:
            low.append(y[i2] - err[i2])
            high.append(y[i2] + err[i2])
            i2 += 1
        self.add_errorbar_series(name, xcats, y, low, high, color, line_width, cap_width_px)

    fn add_eventplot_series_horizontal(
        mut self,
        name: String,
        lanes: List[String],
        events: List[List[Int]],
        color: String = "#5470C6",
        tick_len_px: Int = 10,
        line_width: Int = 2,
        opacity: Int = 100
    ):
        # y axis: lanes as categories
        var lanes_json = String("[")
        var i = 0
        while i < len(lanes):
            lanes_json += _json_str(lanes[i])
            if i + 1 < len(lanes): lanes_json += String(",")
            i += 1
        lanes_json += String("]")

        self.inner.set_yaxis_json(
            String("{ 'type':'category', 'name':'Lane', 'nameLocation':'middle', 'nameGap':40, 'axisLabel':{ 'color':'#333' }, 'data':") +
            lanes_json + String(" }")
        )
        # x axis: numeric
        self.inner.set_xaxis_json(String("{ 'type':'value', 'name':'Value', 'nameLocation':'middle', 'nameGap':30, 'axisLabel':{ 'color':'#333' } }"))

        # data: [x, laneIndex]
        var data = String("[")
        var li = 0
        while li < len(lanes) and li < len(events):
            var ev = events[li]
            var ej = 0
            while ej < len(ev):
                data += String("[") + String(ev[ej]) + String(",") + String(li) + String("]")
                if li < len(lanes) - 1 or ej < len(ev) - 1: data += String(",")
                ej += 1
            li += 1
        data += String("]")

        # renderItem: short vertical tick at each event
        var render = String(
            "function(params, api){" +
            "  var x = api.value(0), lane = api.value(1);" +
            "  var p = api.coord([x, lane]);" +
            "  var half = " + String(tick_len_px) + String(" / 2;" ) +
            "  return {" +
            "    type:'line'," +
            "    shape:{x1:p[0], y1:p[1]-half, x2:p[0], y2:p[1]+half}," +
            "    style:{stroke:'" + color + "', lineWidth:" + String(line_width) + ", opacity:" + String(opacity) + String(" / 100}" ) +
            "  };" +
            "}"
        )

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'custom'")
        obj += String(", 'data':") + data
        obj += String(", 'encode':{ 'x':0, 'y':1 }")
        obj += String(", 'renderItem': ") + render
        obj += String(", 'tooltip':{ 'trigger':'item', 'formatter':") +
               String("function(p){ var v=p.value; return '<b>'+p.name+'</b><br/>Lane: '+v[1]+'<br/>Value: '+v[0]; }") +
               String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)

    fn add_hexbin_series_from_bins(
        mut self,
        name: String,
        centers_x: List[Int],
        centers_y: List[Int],
        values: List[Int],
        radius_px: Int = 14,
        border_width: Int = 1,
        border_color: String = "#333",
        fill_color: String = "#5470C6",
        opacity: Int = 100
    ):
        var n = len(centers_x)
        if len(centers_y) < n: n = len(centers_y)
        if len(values)     < n: n = len(values)

        var data = String("[")
        var i = 0
        while i < n:
            data += String("[") + String(centers_x[i]) + String(",") + String(centers_y[i]) + String(",") + String(values[i]) + String("]")
            if i + 1 < n: data += String(",")
            i += 1
        data += String("]")
 
        var render = String(
            "function(params, api){" +
            "  var cx = api.value(0), cy = api.value(1), val = api.value(2);" +
            "  var p = api.coord([cx, cy]);" +
            "  var R = " + String(radius_px) + String(";" ) +
            "  var a60 = Math.PI/3;" +
            "  var pts = [];" +
            "  for(var k=0;k<6;k++){" +
            "    var ang = a60*k - Math.PI/6;" +
            "    pts.push([ p[0] + R*Math.cos(ang), p[1] + R*Math.sin(ang) ]);" +
            "  }" +
            "  return {" +
            "    type:'polygon'," +
            "    shape:{ points: pts }," +
            "    style: api.style({ fill:'" + fill_color + "', stroke:'" + border_color + "', lineWidth:" + String(border_width) + ", opacity:" + String(opacity) + String(" / 100 }),") +
            "    styleEmphasis:{ opacity:1 }" +
            "  };" +
            "}"
        )
 
        var obj = String("{ 'name':")
        obj += _json_str(name)
        obj += String(", 'type':'custom'")
        obj += String(", 'data':") + data
        obj += String(", 'encode':{ 'x':0, 'y':1, 'value':2 }")
        obj += String(", 'renderItem': ") + render
        obj += String(", 'tooltip':{ 'trigger':'item', 'formatter':")
        obj += String("function(p){ var v=p.value; return '<b>(' + v[0] + ', ' + v[1] + ')</b><br/>Value: <b>' + v[2] + '</b>'; }")
        obj += String(" }")
        obj += String(" }")
 
        self.inner.set_xaxis_json(String("{ 'type':'value', 'name':'X', 'nameLocation':'middle', 'nameGap':30, 'axisLabel':{ 'color':'#333' } }"))
        self.inner.set_yaxis_json(String("{ 'type':'value', 'name':'Y', 'nameLocation':'middle', 'nameGap':40, 'axisLabel':{ 'color':'#333' } }"))

        self.inner.add_series_json(obj)


    # Quiver (vector field) via ECharts custom series on cartesian2d.
    # Data layout per point: [x, y, u, v]
    # magnitude_scale multiplies (u,v) before drawing; head_px sets arrow head size in pixels.
    fn add_quiver_series(mut self,
                         name: String,
                         xs: List[Int],
                         ys: List[Int],
                         us: List[Int],
                         vs: List[Int],
                         color: String = "#5470C6",
                         line_width: Int = 2,
                         head_px: Int = 8,
                         magnitude_scale: Int = 1):
        # Build data: [[x,y,u,v], ...]
        var n = len(xs)
        var data_json = String("[")
        var i = 0
        while i < n and i < len(ys) and i < len(us) and i < len(vs):
            data_json += String("[") + String(xs[i]) + String(",") + String(ys[i]) + String(",") + String(us[i]) + String(",") + String(vs[i]) + String("]")
            if (i + 1) < n and (i + 1) < len(ys) and (i + 1) < len(us) and (i + 1) < len(vs):
                data_json += String(",")
            i += 1
        data_json += String("]")

        # Build the custom series JSON with a renderItem that draws an arrow (line + polygon head).
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'custom', 'coordinateSystem':'cartesian2d'")
        obj += String(", 'encode':{ 'x':0, 'y':1 }")
        obj += String(", 'data':") + data_json

        # Inject the JS renderItem function (unquoted, raw JS).
        obj += String(", 'renderItem': function(params, api) {") +
               String("var x = api.value(0), y = api.value(1), u = api.value(2), v = api.value(3);") +
               String("var scale = ") + String(magnitude_scale) + String(";") +
               String("var start = api.coord([x, y]);") +
               String("var end = api.coord([x + u*scale, y + v*scale]);") +
               String("var dx = end[0] - start[0], dy = end[1] - start[1];") +
               String("var ang = Math.atan2(dy, dx);") +
               String("var head = ") + String(head_px) + String(";") +
               String("var stroke = ") + _json_str(color) + String(";") +
               String("var lw = ") + String(line_width) + String(";") +
               # group with line + triangular head
               String("return { type:'group', children:[") +
               String("{ type:'line', shape:{ x1:start[0], y1:start[1], x2:end[0], y2:end[1] }, ") +
               String("  style:{ stroke:stroke, lineWidth:lw } },") +
               String("{ type:'polygon', shape:{ points:[") +
               String("[ end[0], end[1] ],") +
               String("[ end[0] - head*Math.cos(ang - Math.PI/6), end[1] - head*Math.sin(ang - Math.PI/6) ],") +
               String("[ end[0] - head*Math.cos(ang + Math.PI/6), end[1] - head*Math.sin(ang + Math.PI/6) ]") +
               String("] }, style:{ fill:stroke, stroke:stroke } }") +
               String("] };") +
               String(" }")

        obj += String(" }")

        self.inner.add_series_json(obj)


    # Sankey series
    # nodes_names: unique node names
    # links_*: parallel lists defining edges from source -> target with a value
    fn add_sankey_series(mut self,
                         name: String,
                         nodes_names: List[String],
                         links_source: List[String],
                         links_target: List[String],
                         links_value: List[Int],
                         orient: String = "horizontal",    # 'horizontal' | 'vertical'
                         node_align: String = "justify",    # 'left' | 'right' | 'justify'
                         node_gap: Int = 8,                 # px
                         node_width: Int = 20,              # px
                         draggable: Bool = True,
                         label_show: Bool = True,
                         label_pos: String = "right",
                         edge_color_mode: String = "source",# 'source' | 'target'
                         edge_width: Int = 1,
                         curveness_pct: Int = 33,           # 0..100 (converted to 0..1)
                         opacity: Int = 100,                # 0..100
                         layout_iterations: Int = 32,
                         emphasis_focus: String = "adjacency"):
        # Build nodes
        var nodes_json = String("[")
        var i = 0
        while i < len(nodes_names):
            nodes_json += String("{ 'name':") + _json_str(nodes_names[i]) + String(" }")
            if i + 1 < len(nodes_names): nodes_json += String(",")
            i += 1
        nodes_json += String("]")

        # Build links
        var links_json = String("[")
        var n = len(links_source)
        var j = 0
        while j < n and j < len(links_target) and j < len(links_value):
            links_json += String("{ 'source':") + _json_str(links_source[j]) +
                          String(", 'target':") + _json_str(links_target[j]) +
                          String(", 'value':") + String(links_value[j]) + String(" }")
            if j + 1 < n and j + 1 < len(links_target) and j + 1 < len(links_value):
                links_json += String(",")
            j += 1
        links_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'sankey'")

        # Layout & nodes
        obj += String(", 'orient':") + _json_str(orient)
        obj += String(", 'nodeAlign':") + _json_str(node_align)
        obj += String(", 'nodeGap':") + String(node_gap)
        obj += String(", 'nodeWidth':") + String(node_width)
        obj += String(", 'data':") + nodes_json
        obj += String(", 'links':") + links_json

        # Interactions & style
        obj += String(", 'draggable':") + (String("true") if draggable else String("false"))
        if label_show:
            obj += String(", 'label':{ 'show':true, 'position':") + _json_str(label_pos) + String(" }")
        obj += String(", 'lineStyle':{ 'color':") + _json_str(edge_color_mode) +
               String(", 'width':") + String(edge_width) +
               String(", 'curveness':") + String(curveness_pct) + String(" / 100") +
               String(", 'opacity':") + String(opacity) + String(" / 100 }")
        obj += String(", 'emphasis':{ 'focus':") + _json_str(emphasis_focus) + String(" }")
        obj += String(", 'layoutIterations':") + String(layout_iterations)

        obj += String(" }")
        self.inner.add_series_json(obj)


    # ---- Polar (raw key injection via _extra_kv) -----------------------------

    fn set_polar(mut self,
                 center_x: String = "50%",
                 center_y: String = "55%",
                 radius_pct: Int = 80):
        var obj = String("{ 'center':[ ") + _json_str(center_x) + String(", ") + _json_str(center_y) +
                  String(" ], 'radius':") + _json_str(String(radius_pct) + String("%")) + String(" }")
        # inject: 'polar': { ... }
        self._extra_kv.append(String("'polar': ") + obj)

 

    fn set_angle_axis_value(mut self,
                            min_deg: Int = 0,
                            max_deg: Int = 360,
                            start_angle_deg: Int = 90,
                            clockwise: Bool = True,
                            axis_label_color: String = "#333"):
        var obj = String("{ 'type':'value', 'min':") + String(min_deg) +
                  String(", 'max':") + String(max_deg) +
                  String(", 'startAngle':") + String(start_angle_deg) +
                  String(", 'clockwise':") + (String("true") if clockwise else String("false")) +
                  String(", 'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" } }")
        self._extra_kv.append(String("'angleAxis': ") + obj)

    fn set_radius_axis_value(mut self,
                             name: String = "",
                             min_val: Int = 0,
                             max_val: Int = 400,
                             axis_label_color: String = "#333"):
        var obj = String("{ 'type':'value'")
        if len(name) > 0:
            obj += String(", 'name':") + _json_str(name)
            obj += String(", 'nameLocation':'middle', 'nameGap':28")
        obj += String(", 'min':") + String(min_val)
        obj += String(", 'max':") + String(max_val)
        obj += String(", 'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" } }")
        self._extra_kv.append(String("'radiusAxis': ") + obj)

    # ---- Polar Series (coordinateSystem='polar') -----------------------------
 

    fn add_polar_scatter_series(mut self,
                                name: String,
                                values: List[Int],
                                color: String,
                                symbol_size: Int = 8,
                                opacity: Int = 90):
        var data_json = String("[")
        var i = 0
        while i < len(values):
            data_json += String("[") + String(i) + String(",") + String(values[i]) + String("]")
            if i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")
        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'scatter', 'coordinateSystem':'polar', 'data':") + data_json
        obj += String(", 'symbolSize':") + String(symbol_size)
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(", 'opacity':") + String(opacity) + String(" / 100 }")
        obj += String(" }")
        self.inner.add_series_json(obj)


    # Polar coordinate system helpers

    fn set_angle_axis_category_json(mut self,
                                    categories: List[String],
                                    start_angle_deg: Int = 90,
                                    clockwise: Bool = True,
                                    axis_label_color: String = "#333"):
        var obj = String("{ 'type':'category', ") +
                  String("'startAngle':") + String(start_angle_deg) + String(", ") +
                  String("'clockwise':") + (String("true") if clockwise else String("false")) + String(", ") +
                  String("'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" }, ") +
                  String("'data':") + _json_array_str(categories) +
                  String(" }")
        self.inner.set_angle_axis(obj)   # <-- name fixed

    fn set_radius_axis_value_json(mut self,
                                  name: String = "",
                                  minv: Int = 0,
                                  maxv: Int = 100,
                                  axis_label_color: String = "#333"):
        var obj = String("{ 'type':'value', ") +
                  String("'min':") + String(minv) + String(", ") +
                  String("'max':") + String(maxv) + String(", ") +
                  String("'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" }")
        if len(name) > 0:
            obj += String(", 'name':") + _json_str(name) + String(", 'nameLocation':'end'")
        obj += String(" }")
        self.inner.set_radius_axis(obj)  # <-- name fixed

 

    # Keep only ONE definition of this function
 

    fn add_polar_line_series(mut self,
                            name: String,
                            values: List[Int],
                            color: String,
                            smooth: Bool = False,
                            show_symbol: Bool = True,
                            line_width: Int = 2,
                            area_opacity: Int = 0,
                            show_labels: Bool = False,
                            label_fmt: String = "{c}"):
        var obj = String("{ 'name':") + _json_str(name) +
                String(", 'type':'line', 'coordinateSystem':'polar', 'data':") + _json_array_int(values)
        if smooth: obj += String(", 'smooth':true")
        if show_symbol == False: obj += String(", 'showSymbol':false")
        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")
        if area_opacity > 0:
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")
        self.inner.add_series_json(obj)


    # Tooltip for polar scatter (item trigger)
    fn set_tooltip_polar_item(mut self,
                              bg: String = "#fff",
                              border: String = "#aaa",
                              border_width: Int = 1,
                              text_color: String = "#000",
                              formatter_js: String = ""):
        var tip = String("{ 'trigger':'item', ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)

    # Configure polar axes (value-value). Includes polar container (center+radius).
    fn set_polar_axes_value(mut self,
                            angle_min: Int = 0, angle_max: Int = 360,
                            start_angle: Int = 0, clockwise: Bool = True,
                            radius_min: Int = 0, radius_max: Int = 100,
                            center_x: String = "50%", center_y: String = "55%",
                            radius_pct: Int = 70):
        var ang = String("{ 'type':'value', 'min':") + String(angle_min) +
                  String(", 'max':") + String(angle_max) +
                  String(", 'startAngle':") + String(start_angle) +
                  String(", 'clockwise':") + (String("true") if clockwise else String("false")) +
                  String(" }")
        var rad = String("{ 'type':'value', 'min':") + String(radius_min) +
                  String(", 'max':") + String(radius_max) +
                  String(" }")
        var pol = String("{ 'center':[ ") + _json_str(center_x) + String(", ") + _json_str(center_y) +
                  String(" ], 'radius':") + _json_str(String(radius_pct) + String("%")) + String(" }")

        # Expect *_json setters in Option; rename here if your API differs.
        self.inner.set_angle_axis(ang)
        self.inner.set_radius_axis(rad)
        self.inner.set_polar(pol)

    # Polar scatter series: data as [radius, angle] pairs (r, θ°)
    fn add_polar_scatter_series(mut self,
                                name: String,
                                theta_deg: List[Int],
                                radius_vals: List[Int],
                                color: String,
                                symbol_size: Int = 8,
                                opacity: Int = 100,      # 0..100
                                show_labels: Bool = False,
                                label_fmt: String = "({c})"):
        var data_json = String("[")
        var n = len(theta_deg)
        var i = 0
        while i < n and i < len(radius_vals):
            # ECharts polar expects [radius, angle]
            data_json += String("[") + String(radius_vals[i]) + String(",") + String(theta_deg[i]) + String("]")
            if i + 1 < n and i + 1 < len(radius_vals): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'scatter', 'coordinateSystem':'polar', 'data':") + data_json +
                  String(", 'symbolSize':") + String(symbol_size) +
                  String(", 'itemStyle':{ 'color':") + _json_str(color) +
                  String(", 'opacity':") + String(opacity) + String(" / 100 }")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)


    # --------------------------
    # Polar coordinate utilities
    # --------------------------

 
    # Angle axis as categories (e.g., months around the circle)
    fn set_angle_axis_category(mut self,
                            labels: List[String],
                            start_angle_deg: Int = 90,
                            clockwise: Bool = True,
                            axis_label_color: String = "#333"):
        var obj = String("{ 'type':'category', 'startAngle':") + String(start_angle_deg) +
                String(", 'clockwise':") + (String("true") if clockwise else String("false")) +
                String(", 'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" }, ") +
                String("'data':") + _json_array_str(labels) + String(" }")
        # BEFORE: self.inner.set_angle_axis_json(obj)
        self.inner.set_angle_axis(obj)

    # Radius axis as values (bar length)
    fn set_radius_axis_value(mut self,
                            minv: Int = 0,
                            axis_label_color: String = "#333"):
        var obj = String("{ 'type':'value', 'min':") + String(minv) +
                String(", 'axisLabel':{ 'color':") + _json_str(axis_label_color) + String(" } }")
        # BEFORE: self.inner.set_radius_axis_json(obj)
        self.inner.set_radius_axis(obj)


    # Polar Bar series (data is aligned with angle-axis categories)
    fn add_polar_bar_series(mut self,
                            name: String,
                            data: List[Int],
                            color: String,
                            stack: String = "",           # non-empty => stacked
                            show_labels: Bool = True,
                            label_fmt: String = "{c}",
                            round_cap: Bool = False,
                            bar_width_px: Int = -1):
        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'bar', 'coordinateSystem':'polar', 'data':") + _json_array_int(data)
        if len(stack) > 0:
            obj += String(", 'stack':") + _json_str(stack)
        if round_cap:
            obj += String(", 'roundCap':true")
        if bar_width_px > -1:
            obj += String(", 'barWidth':") + String(bar_width_px)
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'outside', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")
        obj += String(" }")
        self.inner.add_series_json(obj)



  # 3D axes
    fn set_xaxis3d_value(mut self, name: String, name_gap: Int = 20, axis_color: String = "#999"):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLine':{ 'lineStyle':{ 'color':") + _json_str(axis_color) + String(" } } }")
        self.inner.set_xaxis3d_json(obj)

    fn set_yaxis3d_value(mut self, name: String, name_gap: Int = 20, axis_color: String = "#999"):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLine':{ 'lineStyle':{ 'color':") + _json_str(axis_color) + String(" } } }")
        self.inner.set_yaxis3d_json(obj)

    fn set_zaxis3d_value(mut self, name: String, name_gap: Int = 20, axis_color: String = "#999"):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLine':{ 'lineStyle':{ 'color':") + _json_str(axis_color) + String(" } } }")
        self.inner.set_zaxis3d_json(obj)

    # Grid3D
    fn set_grid3d_default(mut self,
                          box_width: Int = 100,
                          box_depth: Int = 100,
                          view_alpha: Int = 25,
                          view_beta: Int = -15,
                          env: String = "auto",
                          post_effect: Bool = False):
        var obj = String("{ 'boxWidth':") + String(box_width) +
                  String(", 'boxDepth':") + String(box_depth) +
                  String(", 'environment':") + _json_str(env) +
                  String(", 'viewControl':{ 'alpha':") + String(view_alpha) +
                  String(", 'beta':") + String(view_beta) + String(" }")
        if post_effect:
            obj += String(", 'postEffect':{ 'enable':true }")
        obj += String(" }")
        self.inner.set_grid3d_json(obj)

    # Surface series (type 'surface'; data: [[x,y,z], ...])
    fn add_surface3d_series(mut self,
                            name: String,
                            xs: List[Int],
                            ys: List[Int],
                            zs: List[Int],
                            wireframe: Bool = False,
                            shading: String = "color"):
        var n = len(xs); var m = len(ys); var k = len(zs)
        var L = n
        if m < L: L = m
        if k < L: L = k

        var data_json = String("[")
        var i = 0
        while i < L:
            data_json += String("[") + String(xs[i]) + String(",") + String(ys[i]) + String(",") + String(zs[i]) + String("]")
            if i + 1 < L: data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'surface', 'data':") + data_json +
                  String(", 'shading':") + _json_str(shading)
        if wireframe:
            obj += String(", 'wireframe':{ 'show':true }")
        obj += String(" }")

        self.inner.add_series_json(obj)


            # Calendar root option injector (single or array)
    fn set_calendar_json(mut self, calendar_json: String):
        # Expects a valid JSON object or array for the 'calendar' option.
        self.add_extra_kv(String("'calendar': ") + calendar_json)

    # Convenient single-calendar builder (horizontal year view)
    fn set_calendar_simple(mut self,
                           range_start: String,       # e.g. "2025-01-01"
                           range_end: String,         # e.g. "2025-12-31"
                           cell_w: Int = 16,
                           cell_h: Int = 16,
                           top_px: Int = 160,
                           left: String = "center",
                           orient: String = "horizontal",
                           year_label_show: Bool = False):
        var cal = String("{ 'range':[ ") + _json_str(range_start) + String(", ") + _json_str(range_end) + String(" ], ") +
                  String("'cellSize':[ ") + String(cell_w) + String(", ") + String(cell_h) + String(" ], ") +
                  String("'left':") + _json_str(left) + String(", 'top':") + String(top_px) + String(", ") +
                  String("'orient':") + _json_str(orient) + String(", ") +
                  String("'yearLabel':{ 'show':") + (String("true") if year_label_show else String("false")) + String(" }, ") +
                  String("'dayLabel':{ 'firstDay':1, 'nameMap':['Sun','Mon','Tue','Wed','Thu','Fri','Sat'] }, ") +
                  String("'monthLabel':{ 'nameMap':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] } ") +
                  String("}")
        self.set_calendar_json(cal)

    # Heatmap series on calendar coordinate system
    fn add_calendar_heatmap_series(mut self,
                                   name: String,
                                   dates: List[String],   # ISO dates "YYYY-MM-DD"
                                   values: List[Int],
                                   calendar_index: Int = 0,
                                   border_width: Int = 0,
                                   border_color: String = "#fff"):
        # data = [ ['YYYY-MM-DD', value], ... ]
        var data_json = String("[")
        var n = len(dates)
        var i = 0
        while i < n and i < len(values):
            data_json += String("[") + _json_str(dates[i]) + String(", ") + String(values[i]) + String("]")
            if i + 1 < n and i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'heatmap', ") +
                  String("'coordinateSystem':'calendar', 'calendarIndex':") + String(calendar_index) +
                  String(", 'data':") + data_json +
                  String(", 'itemStyle':{ 'borderWidth':") + String(border_width) +
                  String(", 'borderColor':") + _json_str(border_color) + String(" } }")
        self.inner.add_series_json(obj)



            # effectScatter series on cartesian2d (value-value axes).
    # xs, ys: same-length lists of points; data -> [[x,y], ...]
    fn add_effect_scatter_series(mut self,
                                 name: String,
                                 xs: List[Int],
                                 ys: List[Int],
                                 color: String,
                                 symbol_size: Int = 12,
                                 opacity: Int = 90,               # 0..100
                                 ripple_scale: Int = 2,            # pulse size multiplier
                                 ripple_brush: String = "stroke",  # 'stroke' | 'fill'
                                 ripple_period_ms: Int = 400,      # lower=faster
                                 show_effect_on_render: Bool = True,
                                 hover_animation: Bool = True,
                                 show_labels: Bool = False,
                                 label_fmt: String = "({c})"):
        # build data array [[x,y], ...]
        var data_json = String("[")
        var n = len(xs)
        var i = 0
        while i < n and i < len(ys):
            data_json += String("[") + String(xs[i]) + String(",") + String(ys[i]) + String("]")
            if i + 1 < n and i + 1 < len(ys): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'effectScatter', 'coordinateSystem':'cartesian2d'")
        obj += String(", 'data':") + data_json
        obj += String(", 'symbolSize':") + String(symbol_size)
        obj += String(", 'hoverAnimation':") + (String("true") if hover_animation else String("false"))
        obj += String(", 'showEffectOn':") + _json_str(String("render") if show_effect_on_render else String("emphasis"))
        obj += String(", 'rippleEffect':{ 'brushType':") + _json_str(ripple_brush) +
               String(", 'scale':") + String(ripple_scale) +
               String(", 'period':") + String(ripple_period_ms) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(", 'opacity':") + String(opacity) + String(" / 100 }")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)


    # PictorialBar series (category x, value y).
    # symbol:
    #   - 'rect', 'circle', 'roundRect', 'triangle', 'diamond', 'pin', 'arrow'
    #   - or a custom SVG path: 'path://M0,10 L10,10 L5,0 Z'
    fn add_pictorial_bar_series(mut self,
                                name: String,
                                data: List[Int],
                                color: String,
                                symbol: String = "rect",
                                symbol_size_w: Int = 24,
                                symbol_size_h: Int = 12,
                                symbol_offset_x: Int = 0,
                                symbol_offset_y: Int = 0,
                                symbol_repeat: String = "auto",   # 'auto' | 'fixed' | 'true' (stacked)
                                symbol_clip: Bool = False,
                                symbol_position: String = "start", # 'start' | 'end' | 'center'
                                zlevel: Int = 0,
                                opacity: Int = 100,
                                show_labels: Bool = False,
                                label_fmt: String = "{c}"):

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'pictorialBar'")
        obj += String(", 'symbol':") + _json_str(symbol)
        obj += String(", 'symbolSize':[") + String(symbol_size_w) + String(",") + String(symbol_size_h) + String("]")
        obj += String(", 'symbolOffset':[") + String(symbol_offset_x) + String(",") + String(symbol_offset_y) + String("]")
        # symbolRepeat: allow 'auto' or explicit text; if you pass 'true' use raw true
        if symbol_repeat == "true":
            obj += String(", 'symbolRepeat':true")
        else:
            obj += String(", 'symbolRepeat':") + _json_str(symbol_repeat)
        if symbol_clip:
            obj += String(", 'symbolClip':true")
        obj += String(", 'symbolPosition':") + _json_str(symbol_position)
        obj += String(", 'zlevel':") + String(zlevel)
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(", 'opacity':") + String(opacity) + String(" / 100 }")

        # labels (optional)
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")

        # data array (numbers only)
        obj += String(", 'data':") + _json_array_int(data)
        obj += String(" }")

        self.inner.add_series_json(obj)


    # Single Axis (time) for ThemeRiver. Injects a 'singleAxis' object via extra KV.
    fn set_single_axis_time(mut self,
                            axis_name: String = "",
                            name_location: String = "middle",
                            name_gap: Int = 25,
                            split_line: Bool = True,
                            boundary_gap: Bool = False,
                            label_color: String = "#333"):
        var obj = String("{ 'type':'time'")
        if len(axis_name) > 0:
            obj += String(", 'name':") + _json_str(axis_name)
            obj += String(", 'nameLocation':") + _json_str(name_location)
            obj += String(", 'nameGap':") + String(name_gap)
        if split_line:
            obj += String(", 'splitLine':{ 'show':true }")
        if boundary_gap:
            obj += String(", 'boundaryGap':true")
        else:
            obj += String(", 'boundaryGap':false")
        obj += String(", 'axisLabel':{ 'color':") + _json_str(label_color) + String(" }")
        obj += String(" }")
        # Inject as top-level 'singleAxis'
        self._extra_kv.append(String("'singleAxis': ") + obj)

    # ThemeRiver series.
    # - names: legend entries (also appear as the 3rd element in data triples)
    # - data triples: [[timestamp_or_date_string, value, name], ...]
    #   Example date: '2025-01-01' or full timestamp '2025-01-01 12:00:00'
    fn add_theme_river_series(mut self,
                              names: List[String],
                              dates: List[String],
                              values: List[Int],
                              group_name: String) -> None:
        # Build [[date, value, name], ...]
        var data_json = String("[")
        var n = len(dates)
        var i = 0
        while i < n and i < len(values):
            data_json += String("[") + _json_str(dates[i]) + String(",") + String(values[i]) + String(",") + _json_str(group_name) + String("]")
            if i + 1 < n and i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'type':'themeRiver', 'data':") + data_json
        obj += String(", 'emphasis':{ 'focus':'series' }")
        obj += String(" }")
        self.inner.add_series_json(obj)


    # Area series (line + areaStyle). Supports stacking via `stack` name.
    fn add_area_series(mut self,
                       name: String,
                       data: List[Int],
                       color: String,
                       smooth: Bool = False,
                       show_symbol: Bool = False,
                       line_width: Int = 2,
                       area_opacity: Int = 40,     # 0..100 (0 disables fill)
                       stack: String = ""):
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'line', 'data':") + _json_array_int(data)

        if len(stack) > 0:
            obj += String(", 'stack':") + _json_str(stack)

        if smooth:
            obj += String(", 'smooth':true")

        if show_symbol == False:
            obj += String(", 'showSymbol':false")

        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")

        if area_opacity > 0:
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")

        obj += String(" }")
        self.inner.add_series_json(obj)


        # Candlestick series for financial OHLC data.
    # ECharts expects data points as [open, close, low, high].
    fn add_candlestick_series(mut self,
                              name: String,
                              opens: List[Int],
                              closes: List[Int],
                              lows: List[Int],
                              highs: List[Int],
                              up_color: String = "#ec0000",       # rising candle fill
                              down_color: String = "#00da3c",     # falling candle fill
                              up_border: String = "#8A0000",
                              down_border: String = "#008F28"):
        # Build [[o,c,l,h], ...]
        var n = len(opens)
        var data_json = String("[")
        var i = 0
        while i < n and i < len(closes) and i < len(lows) and i < len(highs):
            data_json += String("[") + String(opens[i]) + String(",") +
                                   String(closes[i]) + String(",") +
                                   String(lows[i])   + String(",") +
                                   String(highs[i])  + String("]")
            if i + 1 < n and i + 1 < len(closes) and i + 1 < len(lows) and i + 1 < len(highs):
                data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'candlestick', 'data':") + data_json
        obj += String(", 'itemStyle':{ ") +
               String("'color':") + _json_str(up_color) + String(", ") +
               String("'color0':") + _json_str(down_color) + String(", ") +
               String("'borderColor':") + _json_str(up_border) + String(", ") +
               String("'borderColor0':") + _json_str(down_border) +
               String(" }")

        # (Optional) You can add markPoint/markLine via add_series_json extras if needed.

        obj += String(" }")
        self.inner.add_series_json(obj)


    # Funnel series
    # - labels: category names (each is a segment)
    # - values: numeric values (same length as labels)
    # - layout: control position & size (left/top/width/height) in percents or px
    # - sort: 'descending' (default), 'ascending', or 'none'
    # - minSize/maxSize: e.g. 0..100 (percentage of container width/height based on orient)
    fn add_funnel_series(mut self,
                         name: String,
                         labels: List[String],
                         values: List[Int],
                         left: String = "10%",
                         top: String = "20%",
                         width: String = "80%",
                         height: String = "70%",
                         min_value: Int = 0,
                         max_value: Int = 100,
                         min_size_pct: Int = 0,     # 0 -> '0%'
                         max_size_pct: Int = 100,   # 100 -> '100%'
                         sort: String = "descending",   # 'descending' | 'ascending' | 'none'
                         gap_px: Int = 2,
                         label_show: Bool = True,
                         label_position: String = "outside",
                         label_formatter: String = "{b}: {c} ({d}%)",  # {d} is not default in funnel; ECharts computes percent only in pie. We still keep it customizable.
                         item_opacity: Int = 100,
                         border_color: String = "#fff",
                         border_width: Int = 1):
        # Build data array: [{name:'A', value:10}, ...]
        var data_json = String("[")
        var n = len(labels)
        var i = 0
        while i < n and i < len(values):
            data_json += String("{ 'name':") + _json_str(labels[i]) +
                         String(", 'value':") + String(values[i]) + String(" }")
            if i + 1 < n and i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'funnel'")
        obj += String(", 'left':")   + _json_str(left)
        obj += String(", 'top':")    + _json_str(top)
        obj += String(", 'width':")  + _json_str(width)
        obj += String(", 'height':") + _json_str(height)

        obj += String(", 'min':") + String(min_value)
        obj += String(", 'max':") + String(max_value)
        obj += String(", 'minSize':") + _json_str(String(min_size_pct) + String("%"))
        obj += String(", 'maxSize':") + _json_str(String(max_size_pct) + String("%"))

        if len(sort) > 0:
            obj += String(", 'sort':") + _json_str(sort)

        obj += String(", 'gap':") + String(gap_px)

        # Labels
        if label_show:
            obj += String(", 'label':{ 'show':true, 'position':") + _json_str(label_position) +
                   String(", 'formatter':") + _json_str(label_formatter) + String(" }")

        # Item style
        obj += String(", 'itemStyle':{ 'opacity':") + String(item_opacity) +
               String(" / 100, 'borderColor':") + _json_str(border_color) +
               String(", 'borderWidth':") + String(border_width) + String(" }")

        obj += String(", 'data':") + data_json
        obj += String(" }")

        self.inner.add_series_json(obj)


    # Gauge series
    # - value: current value to display
    # - minv/maxv: gauge range
    # - unit: appended in detail formatter
    # - start_angle/end_angle: degrees (ECharts default: 225 -> -45)
    # - radius_pct: e.g., 80 => '80%'
    # - axisline segments: three-color thresholds (t1,t2 in 0..1)
    fn add_gauge_series(mut self,
                        name: String,
                        value: Int,
                        minv: Int = 0,
                        maxv: Int = 100,
                        unit: String = "%",
                        start_angle: Int = 225,
                        end_angle: Int = -45,
                        radius_pct: Int = 80,
                        center_x: String = "50%",
                        center_y: String = "60%",
                        show_pointer: Bool = True,
                        show_progress: Bool = True,
                        progress_width: Int = 12,
                        axis_width: Int = 12,
                        t1: Int = 30,                # threshold 1 as percent (0..100)
                        t2: Int = 70,                # threshold 2 as percent (0..100)
                        col_low: String = "#91CC75",
                        col_mid: String = "#FAC858",
                        col_high: String = "#EE6666",
                        detail_formatter: String = "{value}%"):

        # convert t1,t2 to 0..1 ratios in JS
        var seg = String("[ [") + String(t1) + String("/100, ") + _json_str(col_low) + String("], ") +
                  String("[") + String(t2) + String("/100, ") + _json_str(col_mid) + String("], ") +
                  String("[1, ") + _json_str(col_high) + String("] ]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'gauge'")
        obj += String(", 'min':") + String(minv)
        obj += String(", 'max':") + String(maxv)
        obj += String(", 'startAngle':") + String(start_angle)
        obj += String(", 'endAngle':") + String(end_angle)
        obj += String(", 'radius':'") + String(radius_pct) + String("%'")
        obj += String(", 'center':[ ") + _json_str(center_x) + String(", ") + _json_str(center_y) + String(" ]")

        if show_progress:
            obj += String(", 'progress':{ 'show':true, 'width':") + String(progress_width) + String(" }")

        obj += String(", 'axisLine':{ 'lineStyle':{ 'width':") + String(axis_width) + String(", 'color':") + seg + String(" } }")

        obj += String(", 'pointer':{ 'show':") + (String("true") if show_pointer else String("false")) + String(" }")

        # Ticks/labels aesthetics (can be tuned)
        obj += String(", 'splitLine':{ 'length':12, 'lineStyle':{ 'width':2 } }")
        obj += String(", 'axisTick':{ 'length':8 }")
        obj += String(", 'axisLabel':{ 'color':'#444' }")

        # Detail box (readout)
        obj += String(", 'detail':{ 'formatter':") + _json_str(detail_formatter) + String(", 'fontSize':18 }")

        # Data: [{value: X, name: '...'}]
        obj += String(", 'data':[ { 'value':") + String(value) + String(", 'name':") + _json_str(name) + String(" } ]")

        obj += String(" }")

        self.inner.add_series_json(obj)

    # Map series (choropleth). 'map_name' must be registered in pre_init_js: echarts.registerMap(map_name, geojson).
    fn add_map_choropleth(mut self,
                          map_name: String,
                          data_labels: List[String],
                          data_values: List[Int],
                          roam: Bool = True,
                          show_labels: Bool = False,
                          label_color: String = "#000"):
        # Build data: [{name:'X', value: N}, ...]
        var data_json = String("[")
        var n = len(data_labels)
        var i = 0
        while i < n and i < len(data_values):
            data_json += String("{ 'name':") + _json_str(data_labels[i]) +
                         String(", 'value':") + String(data_values[i]) + String(" }")
            if i + 1 < n and i + 1 < len(data_values):
                data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'type':'map', 'map':") + _json_str(map_name)
        if roam:
            obj += String(", 'roam':true")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'color':") + _json_str(label_color) + String(" }")
        obj += String(", 'data':") + data_json + String(" }")

        self.inner.add_series_json(obj)


        # --- Parallel coordinates support ---------------------------------------
    # We collect axes & layout locally and inject them at finalize (to_json).
 

    # Call this once in __init__
    # (Add to your __init__ if not already there)
    #   self._parallel_axes = List[String]()
    #   self._parallel_obj = String("")

    fn set_parallel_layout(mut self,
                           left_pct: String = "5%",
                           right_pct: String = "10%",
                           top_px: Int = 160,
                           bottom_px: Int = 60):
        # Build the 'parallel' object (layout/padding/etc.)
        self._parallel_obj = String("{ 'left':") + _json_str(left_pct) +
                             String(", 'right':") + _json_str(right_pct) +
                             String(", 'top':") + String(top_px) +
                             String(", 'bottom':") + String(bottom_px) +
                             String(" }")

    fn add_parallel_axis(mut self,
                         dim: Int,
                         name: String,
                         type_: String = "value",     # 'value' | 'category' | ...
                         min_: String = "",
                         max_: String = ""):
        # Build single parallelAxis object: {dim, name, type, [min], [max]}
        var obj = String("{ 'dim':") + String(dim) +
                  String(", 'name':") + _json_str(name) +
                  String(", 'type':") + _json_str(type_)
        if len(min_) > 0:
            obj += String(", 'min':") + _json_str(min_)
        if len(max_) > 0:
            obj += String(", 'max':") + _json_str(max_)
        obj += String(" }")
        self._parallel_axes.append(obj)

    fn add_parallel_series_int(mut self,
                               name: String,
                               rows: List[List[Int]],
                               color: String = "",
                               line_opacity: Int = 70,      # 0..100
                               smooth: Bool = False,
                               show_symbol: Bool = False):
        # data: [[v0,v1,v2,...],[...],...]
        var data_json = String("[")
        var r = 0
        while r < len(rows):
            var row = rows[r]
            # one row
            data_json += String("[")
            var c = 0
            while c < len(row):
                data_json += String(row[c])
                if c + 1 < len(row): data_json += String(",")
                c += 1
            data_json += String("]")
            if r + 1 < len(rows): data_json += String(",")
            r += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'parallel', 'data':") + data_json
        obj += String(", 'lineStyle':{ 'width':1, 'opacity':") + String(line_opacity) + String(" / 100 }")
        if len(color) > 0:
            obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")
        if smooth:
            obj += String(", 'smooth':true")
        if show_symbol == False:
            obj += String(", 'showSymbol':false")
        obj += String(" }")

        self.inner.add_series_json(obj)

    # Radar coordinate setup
    # Build radar.indicator from parallel lists: names[i], maxes[i] (and optional mins[i])
    fn set_radar_from_names_maxes(mut self,
                                  names: List[String],
                                  maxes: List[Int],
                                  shape: String = "polygon",     # "polygon" or "circle"
                                  split_number: Int = 5,
                                  start_angle_deg: Int = 90,
                                  name_color: String = "#333"):
        var n = len(names)
        var m = len(maxes)
        var k = n if n < m else m
        var ind = String("[")
        var i = 0
        while i < k:
            ind += String("{ 'name':") + _json_str(names[i]) +
                   String(", 'max':") + String(maxes[i]) + String(" }")
            if i + 1 < k: ind += String(",")
            i += 1
        ind += String("]")

        var radar_json = String("{ ") +
                         String("'indicator':") + ind +
                         String(", 'shape':") + _json_str(shape) +
                         String(", 'splitNumber':") + String(split_number) +
                         String(", 'startAngle':") + String(start_angle_deg) +
                         String(", 'name':{ 'color':") + _json_str(name_color) + String(" }") +
                         String(" }")
        # Option is expected to expose set_radar(...) in your library.
        self.inner.set_radar(radar_json)

    # Radar series
    # values: one value per indicator, in the same order as set_radar indicators.
    fn add_radar_series(mut self,
                        name: String,
                        values: List[Int],
                        color: String,
                        line_width: Int = 2,
                        area_opacity: Int = 0,      # 0..100 (0=no area)
                        show_symbol: Bool = True,
                        symbol: String = "circle",
                        symbol_size: Int = 6,
                        show_labels: Bool = False,
                        label_fmt: String = "{c}"):
        var data_arr = _json_array_int(values)
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'radar'")
        obj += String(", 'data':[ { 'value':") + data_arr + String(", 'name':") + _json_str(name) + String(" } ]")
        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")
        if area_opacity > 0:
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")
        if show_symbol == False:
            obj += String(", 'showSymbol':false")
        if len(symbol) > 0:
            obj += String(", 'symbol':") + _json_str(symbol)
        obj += String(", 'symbolSize':") + String(symbol_size)
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")
        self.inner.add_series_json(obj)

    # Tooltip suited for radar (item trigger)
    fn set_tooltip_item_for_radar(mut self,
                                  bg: String = "#fff",
                                  border: String = "#aaa",
                                  border_width: Int = 1,
                                  text_color: String = "#000",
                                  formatter_js: String = ""):
        var tip = String("{ 'trigger':'item', ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)


    # Build a sunburst node as JSON string.
    # - name: node label
    # - value: numeric value (0 allowed; children may carry values)
    # - children: list of child JSON strings (built with sb_node as well)
    fn sb_node(self, name: String, value: Int = 0, children: List[String] = List[String]()) -> String:
        var obj = String("{ 'name':") + _json_str(name)
        if value != 0:
            obj += String(", 'value':") + String(value)
        if len(children) > 0:
            obj += String(", 'children':[")
            var i = 0
            while i < len(children):
                obj += children[i]
                if i + 1 < len(children): obj += String(",")
                i += 1
            obj += String("]")
        obj += String(" }")
        return obj

    # Tooltip tuned for sunburst (item trigger).
    fn set_tooltip_sunburst(mut self,
                            bg: String = "#fff",
                            border: String = "#aaa",
                            border_width: Int = 1,
                            text_color: String = "#000",
                            formatter_js: String = ""):
        var tip = String("{ 'trigger':'item', ") +
                  String(" 'backgroundColor':") + _json_str(bg) + String(",") +
                  String(" 'borderColor':") + _json_str(border) + String(",") +
                  String(" 'borderWidth':") + String(border_width) + String(",") +
                  String(" 'textStyle':{ 'color':") + _json_str(text_color) + String(" }")
        if len(formatter_js) > 0:
            tip += String(", 'formatter': ") + formatter_js
        tip += String(" }")
        self.inner.set_tooltip(tip)

    # Add sunburst series.
    # - roots: list of top-level node JSON strings (from sb_node)
    # - radius_inner_pct > 0 makes a "ring-like" center
    # - levels_json: optional fine-grained level config, pass raw JSON or empty
    fn add_sunburst_series(mut self,
                           name: String,
                           roots: List[String],
                           radius_outer_pct: Int = 90,
                           radius_inner_pct: Int = 0,
                           center_x: String = "50%",
                           center_y: String = "52%",
                           sort: String = "desc",               # 'desc' | 'asc' | 'null'
                           emphasis_focus: String = "ancestor", # 'ancestor' | 'self' | 'none'
                           levels_json: String = String(""),
                           node_click: String = "rootToNode"):  # 'rootToNode' | false
        var data_json = String("[")
        var i = 0
        while i < len(roots):
            data_json += roots[i]
            if i + 1 < len(roots): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'sunburst'")
        obj += String(", 'center':[ ") + _json_str(center_x) + String(", ") + _json_str(center_y) + String(" ]")
        obj += String(", 'radius':[ '") + String(radius_inner_pct) + String("%', '") +
               String(radius_outer_pct) + String("%' ]")
        obj += String(", 'sort':") + ( _json_str(sort) if sort != "null" else String("null") )
        obj += String(", 'emphasis':{ 'focus':") + _json_str(emphasis_focus) + String(" }")
        obj += String(", 'nodeClick':") + ( _json_str(node_click) if node_click != "false" else String("false") )
        obj += String(", 'data':") + data_json
        if len(levels_json) > 0:
            obj += String(", 'levels':") + levels_json
        obj += String(" }")

        self.inner.add_series_json(obj)



    # Build a flat treemap data array from labels and values:
    # Returns JSON like: [ { 'name':'A','value':10 }, ... ]
    fn build_treemap_flat(self, labels: List[String], values: List[Int]) -> String:
        var data_json = String("[")
        var n = len(labels)
        var i = 0
        while i < n and i < len(values):
            data_json += String("{ 'name':") + _json_str(labels[i]) +
                         String(", 'value':") + String(values[i]) + String(" }")
            if i + 1 < n and i + 1 < len(values): data_json += String(",")
            i += 1
        data_json += String("]")
        return data_json

    # Add a Treemap series.
    # - Provide either your own `data_json` (hierarchical) or use `build_treemap_flat(...)`.
    # - `levels_json` lets you pass ECharts levels config (array JSON) for styles/visual mapping.
    fn add_treemap_series(mut self,
                          name: String,
                          data_json: String,
                          leaf_depth: Int = 1,
                          roam: Bool = True,
                          node_click: String = "zoom",     # 'zoom' | 'link' | '' (means false)
                          breadcrumb_show: Bool = True,
                          label_show: Bool = True,
                          upper_label_show: Bool = True,
                          levels_json: String = ""):
        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'treemap'")

        # data (must be a valid JSON array/object string)
        obj += String(", 'data':") + data_json

        # behavior & UI
        obj += String(", 'leafDepth':") + String(leaf_depth)
        obj += String(", 'roam':") + (String("true") if roam else String("false"))

        # nodeClick can be string or false
        if len(node_click) > 0:
            obj += String(", 'nodeClick':") + _json_str(node_click)
        else:
            obj += String(", 'nodeClick':false")

        obj += String(", 'breadcrumb':{ 'show':") + (String("true") if breadcrumb_show else String("false")) + String(" }")

        # labels
        if label_show:
            obj += String(", 'label':{ 'show':true, 'formatter':'{b}' }")
        if upper_label_show:
            obj += String(", 'upperLabel':{ 'show':true, 'formatter':'{b}' }")

        # optional levels (array JSON)
        if len(levels_json) > 0:
            obj += String(", 'levels':") + levels_json

        obj += String(" }")
        self.inner.add_series_json(obj)



        # Build a tree node JSON object from a list of child node JSON strings.
    fn build_tree_node(self,
                       name: String,
                       value: Int = 0,
                       children: List[String] = List[String](),
                       collapsed: Bool = False) -> String:
        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'value':") + String(value)
        # children
        obj += String(", 'children':[")
        var i = 0
        while i < len(children):
            obj += children[i]
            if i + 1 < len(children): obj += String(",")
            i += 1
        obj += String("]")
        # collapsed (initial folded state)
        if collapsed:
            obj += String(", 'collapsed':true")
        obj += String(" }")
        return obj

    # Add a Tree series. 'root' must be a single JSON object (a node).
    # Common options are exposed; you can extend by add_extra_kv if needed.
    fn add_tree_series(mut self,
                       name: String,
                       root: String,
                       orient: String = "LR",          # 'LR' | 'RL' | 'TB' | 'BT'
                       layout: String = "orthogonal",  # 'orthogonal' | 'radial'
                       symbol_size: Int = 10,
                       left: String = "8%",
                       right: String = "8%",
                       top: String = "160px",
                       bottom: String = "10%",
                       label_pos: String = "left",
                       label_align: String = "right",
                       leaves_label_pos: String = "right",
                       leaves_label_align: String = "left",
                       expand_and_collapse: Bool = True,
                       initial_tree_depth: Int = 2,
                       line_curveness_percent: Int = 0  # 0..100
                       ):
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'tree'")
        obj += String(", 'data':[ ") + root + String(" ]")
        obj += String(", 'left':") + _json_str(left) +
               String(", 'right':") + _json_str(right) +
               String(", 'top':") + _json_str(top) +
               String(", 'bottom':") + _json_str(bottom)
        obj += String(", 'symbolSize':") + String(symbol_size)
        obj += String(", 'orient':") + _json_str(orient)
        obj += String(", 'layout':") + _json_str(layout)
        obj += String(", 'label':{ 'position':") + _json_str(label_pos) +
               String(", 'verticalAlign':'middle', 'align':") + _json_str(label_align) + String(" }")
        obj += String(", 'leaves':{ 'label':{ 'position':") + _json_str(leaves_label_pos) +
               String(", 'verticalAlign':'middle', 'align':") + _json_str(leaves_label_align) + String(" } }")
        obj += String(", 'expandAndCollapse':") + (String("true") if expand_and_collapse else String("false"))
        obj += String(", 'initialTreeDepth':") + String(initial_tree_depth)
        obj += String(", 'lineStyle':{ 'width':1, 'curveness':") + String(line_curveness_percent) + String("/100 }")
        obj += String(" }")
        self.inner.add_series_json(obj)


 

    fn add_wordcloud_series(mut self,
                            name: String,
                            words: List[String],
                            weights: List[Int],
                            size_min_px: Int = 12,
                            size_max_px: Int = 48,
                            rotation_min_deg: Int = -90,
                            rotation_max_deg: Int = 90,
                            rotation_step_deg: Int = 45,
                            grid_size: Int = 8,
                            shape: String = "circle",
                            draw_out_of_bound: Bool = False):
        var data_json = String("[")
        var n = len(words); var i = 0
        while i < n and i < len(weights):
            data_json += String("{ 'name':") + _json_str(words[i]) +
                         String(", 'value':") + String(weights[i]) + String(" }")
            if i + 1 < n and i + 1 < len(weights): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'wordCloud'")
        obj += String(", 'gridSize':") + String(grid_size)
        obj += String(", 'sizeRange':[") + String(size_min_px) + String(",") + String(size_max_px) + String("]")
        obj += String(", 'rotationRange':[") + String(rotation_min_deg) + String(",") + String(rotation_max_deg) + String("]")
        obj += String(", 'rotationStep':") + String(rotation_step_deg)
        obj += String(", 'shape':") + _json_str(shape)
        obj += String(", 'drawOutOfBound':") + (String("true") if draw_out_of_bound else String("false"))
        obj += String(", 'textStyle':{ 'color': function(){ var cs=['#5470C6','#91CC75','#FAC858','#EE6666','#73C0DE','#3BA272','#FC8452','#9A60B4','#EA7CCC']; return cs[Math.floor(Math.random()*cs.length)]; } }")
        obj += String(", 'data':") + data_json
        obj += String(" }")

        self.inner.add_series_json(obj)


        # Step Line series: step ∈ {'start','middle','end'}
    fn add_step_line_series(mut self,
                            name: String,
                            data: List[Int],
                            color: String,
                            step: String,                  # 'start' | 'middle' | 'end'
                            show_symbol: Bool = True,
                            line_width: Int = 2,
                            area_opacity: Int = 0,         # 0..100 (0 = no area)
                            show_labels: Bool = False,
                            label_fmt: String = "{c}"):
        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'line', 'data':") + _json_array_int(data) +
                  String(", 'step':") + _json_str(step)

        if show_symbol == False:
            obj += String(", 'showSymbol':false")

        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")

        if area_opacity > 0:
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")

        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")

        obj += String(" }")
        self.inner.add_series_json(obj)


    # Waterfall (cumulative/delta) using stacked bars:
    # We create three stacked bar series:
    #  - 'assist'  : transparent placeholder that positions each delta at previous cumulative
    #  - 'positive': positive deltas only (>=0), labeled on top
    #  - 'negative': negative deltas only (<0),  labeled at bottom
    # Optionally a final total bar is appended.
    fn add_waterfall_series(mut self,
                            categories: List[String],
                            deltas: List[Int],
                            color_positive: String = "#91CC75",
                            color_negative: String = "#EE6666",
                            color_assist: String = "transparent",
                            show_total: Bool = True,
                            total_label: String = "Total",
                            label_fmt_pos: String = "{c}",
                            label_fmt_neg: String = "{c}"):

        # 1) Build arrays
        var n = len(deltas)
        var cats = List[String]()
        var assist = List[Int]()
        var positives = List[Int]()
        var negatives = List[Int]()
        var cum = List[Int]()         # cumulative at each index (after applying delta)

        # Copy categories (truncate to n if longer)
        var i = 0
        while i < n and i < len(categories):
            cats.append(categories[i])
            i += 1

        var prev = 0
        var j = 0
        while j < n:
            # placeholder is previous cumulative
            assist.append(prev)

            var d = deltas[j]
            if d >= 0:
                positives.append(d)
                negatives.append(0)
            else:
                positives.append(0)
                negatives.append(d)  # keep negative sign
            prev = prev + d
            cum.append(prev)
            j += 1

        # Optional: add final total
        if show_total:
            cats.append(total_label)
            assist.append(0)
            positives.append(prev)
            negatives.append(0)
            cum.append(prev)

        # 2) Save cumulative for tooltips (inject as custom KV on the option root)
        #    You can access it in JS formatter via `option._cum`.
        var cum_json = _json_array_int(cum)
        self._extra_kv.append(String("'_cum': ") + cum_json)

        # 3) x/y axes (value-value label styling handled by caller as needed)
        #    NOTE: We only set X categories if the caller hasn't set them already.
        #    If you prefer, feel free to override after calling this function.
        var x_json = String("{ 'type':'category', 'name':'', 'axisLabel':{ 'color':'#333' }, 'data':") +
                     _json_array_str(cats) + String(" }")
        self.inner.set_xaxis_json(x_json)

        var y_json = String("{ 'type':'value', 'name':'', 'axisLabel':{ 'color':'#333' } }")
        self.inner.set_yaxis_json(y_json)

        # 4) Build JSON arrays
        var assist_json = _json_array_int(assist)
        var pos_json    = _json_array_int(positives)
        var neg_json    = _json_array_int(negatives)

        # 5) Add three stacked series
        # Assist (transparent)
        var s_assist = String("{ 'name':'assist','type':'bar','stack':'sum', 'itemStyle':{ 'borderColor':'transparent','color':") +
                      _json_str(color_assist) + String(" }, 'emphasis':{ 'disabled':true }, 'data':") + assist_json + String(" }")
        self.inner.add_series_json(s_assist)

        # Positive deltas
        var s_pos = String("{ 'name':'increase','type':'bar','stack':'sum', 'label':{ 'show':true,'position':'top','formatter':") +
                    _json_str(label_fmt_pos) + String(" }, 'itemStyle':{ 'color':") + _json_str(color_positive) +
                    String(" }, 'data':") + pos_json + String(" }")
        self.inner.add_series_json(s_pos)

        # Negative deltas (labels on bottom)
        var s_neg = String("{ 'name':'decrease','type':'bar','stack':'sum', 'label':{ 'show':true,'position':'bottom','formatter':") +
                    _json_str(label_fmt_neg) + String(" }, 'itemStyle':{ 'color':") + _json_str(color_negative) +
                    String(" }, 'data':") + neg_json + String(" }")
        self.inner.add_series_json(s_neg)


    # Quick helpers for value axes with optional min/max
    fn set_xaxis_value_json(mut self, name: String, name_location: String, name_gap: Int,
                             labels_color: String, minv: String = "", maxv: String = ""):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameLocation':") + _json_str(name_location) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLabel':{ 'color':") + _json_str(labels_color) + String(" }")
        if len(minv) > 0: obj += String(", 'min':") + minv
        if len(maxv) > 0: obj += String(", 'max':") + maxv
        obj += String(" }")
        self.inner.set_xaxis_json(obj)

    fn set_yaxis_value_json2(mut self, name: String, name_location: String, name_gap: Int,
                              labels_color: String, minv: String = "", maxv: String = ""):
        var obj = String("{ 'type':'value', 'name':") + _json_str(name) +
                  String(", 'nameLocation':") + _json_str(name_location) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLabel':{ 'color':") + _json_str(labels_color) + String(" }")
        if len(minv) > 0: obj += String(", 'min':") + minv
        if len(maxv) > 0: obj += String(", 'max':") + maxv
        obj += String(" }")
        self.inner.set_yaxis_json(obj)


    fn set_yaxis_category_json(mut self, name: String, name_location: String, name_gap: Int, labels_color: String, data: List[String]):
        var obj = String("{ 'type':'category', 'name':") + _json_str(name) +
                  String(", 'nameLocation':") + _json_str(name_location) +
                  String(", 'nameGap':") + String(name_gap) +
                  String(", 'axisLabel':{ 'color':") + _json_str(labels_color) + String(" }, ") +
                  String(" 'data':") + _json_array_str(data) + String(" }")
        self.inner.set_yaxis_json(obj)


        # Category axes from label lists (no need to call internal helpers from the demo)
    fn set_xaxis_category_labels(mut self,
                                 labels: List[String],
                                 name: String = "",
                                 name_location: String = "middle",
                                 name_gap: Int = 28,
                                 labels_color: String = "#333",
                                 rotate_deg: Int = 0):
        var obj = String("{ 'type':'category'")
        if len(name) > 0:
            obj += String(", 'name':") + _json_str(name) +
                   String(", 'nameLocation':") + _json_str(name_location) +
                   String(", 'nameGap':") + String(name_gap)
        obj += String(", 'data':") + _json_array_str(labels) +
               String(", 'axisLabel':{ 'color':") + _json_str(labels_color) +
               String(", 'rotate':") + String(rotate_deg) + String(" } }")
        self.inner.set_xaxis_json(obj)

    fn set_yaxis_category_labels(mut self,
                                 labels: List[String],
                                 name: String = "",
                                 name_location: String = "middle",
                                 name_gap: Int = 28,
                                 labels_color: String = "#333",
                                 rotate_deg: Int = 0):
        var obj = String("{ 'type':'category'")
        if len(name) > 0:
            obj += String(", 'name':") + _json_str(name) +
                   String(", 'nameLocation':") + _json_str(name_location) +
                   String(", 'nameGap':") + String(name_gap)
        obj += String(", 'data':") + _json_array_str(labels) +
               String(", 'axisLabel':{ 'color':") + _json_str(labels_color) +
               String(", 'rotate':") + String(rotate_deg) + String(" } }")
        self.inner.set_yaxis_json(obj)


    # Heatmap from regular grid xs, ys, z[H][W]
    fn add_heatmap_series(mut self, name: String,
                          xs: List[Float64], ys: List[Float64],
                          z: List[List[Float64]],
                          blur_size: Int = 0):
        var H = len(ys)
        var W = len(xs)
        var data = String("[")
        var r = 0
        while r < H:
            var c = 0
            while c < W:
                data += String("[") + String(xs[c]) + String(",") + String(ys[r]) + String(",") + String(z[r][c]) + String("]")
                if not (r == H-1 and c == W-1): data += String(",")
                c += 1
            r += 1
        data += String("]")
        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'heatmap', 'data':") + data
        if blur_size > 0:
            obj += String(", 'blurSize':") + String(blur_size)
        obj += String(", 'progressive':0 }")
        self.inner.add_series_json(obj)

    # Contour lines via Marching Squares on regular grid (xs,ys,z) at given levels
    fn add_contour_lines(mut self,
                         name_prefix: String,
                         xs: List[Float64],
                         ys: List[Float64],
                         z: List[List[Float64]],
                         levels: List[Float64],
                         color: String = "#000000",
                         line_width: Int = 1,
                         opacity: Int = 100):
        var H = len(ys)
        var W = len(xs)
        if H < 2 or W < 2:
            return

        var li = 0
        while li < len(levels):
            var lvl = levels[li]
            var data = String("[")
            var first = True

            var r = 0
            while r < H - 1:
                var c = 0
                while c < W - 1:
                    var xL = xs[c]
                    var xR = xs[c+1]
                    var yB = ys[r]
                    var yT = ys[r+1]

                    var z00 = z[r][c]
                    var z10 = z[r][c+1]
                    var z01 = z[r+1][c]
                    var z11 = z[r+1][c+1]

                    var idx = 0
                    if z00 > lvl: idx += 1
                    if z10 > lvl: idx += 2
                    if z01 > lvl: idx += 4
                    if z11 > lvl: idx += 8

                    if idx != 0 and idx != 15:
                        # helper to append one segment (two points) + null separator
                   

                        # compute per-case edge intersections and append
                        if idx == 1 or idx == 14:
                            var a = _edge_point(3, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(0, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 2 or idx == 13:
                            var a = _edge_point(0, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(1, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 3 or idx == 12:
                            var a = _edge_point(3, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(1, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 4 or idx == 11:
                            var a = _edge_point(2, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(3, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 5:
                            # ambiguous: choose (top,bottom)
                            var a = _edge_point(2, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(0, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 6 or idx == 9:
                            var a = _edge_point(2, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(1, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 7 or idx == 8:
                            var a = _edge_point(2, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(1, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])
                        elif idx == 10:
                            # ambiguous: choose (left,right)
                            var a = _edge_point(3, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            var b = _edge_point(1, xL,xR,yB,yT, z00,z10,z01,z11, lvl)
                            _append_segment(data, first, a[0], a[1], b[0], b[1])

                    c += 1
                r += 1

            data += String("]")

            var s = String("{ 'name':") + _json_str(name_prefix + String(" ") + String(lvl)) +
                    String(", 'type':'line', 'data':") + data +
                    String(", 'symbol':'none', 'lineStyle':{ 'width':") + String(line_width) +
                    String(", 'color':") + _json_str(color) +
                    String(", 'opacity':") + String(opacity) + String(" / 100 } }")
            self.inner.add_series_json(s)

            li += 1


    # Build Empirical CDF (ECDF) or Survival(=1-ECDF) from raw samples.
    # - samples: raw observations (integers). You fully control pre-processing upstream.
    # - survival=True yields S(x)=1-ECDF(x).
    # - normalized=True outputs probabilities in [0,1]; False => cumulative counts.
    # - step_style: 'end'|'start'|'middle' (ECharts line.step). Default 'end' (right-continuous ECDF).
    # - show_symbol: show/ hide markers at steps.
    # Notes:
    #   We sort samples, build (x,y) pairs as a step function, and feed as value-value line data [[x,y],...].
    # Replace the entire body of add_ecdf_series with this safe, integer-only implementation.
    fn add_ecdf_series(mut self,
                    name: String,
                    samples: List[Int],
                    color: String,
                    survival: Bool = False,
                    normalized: Bool = True,
                    step_style: String = "end",
                    show_symbol: Bool = False,
                    line_width: Int = 2,
                    area_opacity: Int = 0,
                    show_labels: Bool = False,
                    label_fmt: String = "{c}"):

        # Defensive copy + simple insertion sort (OK for demo-sized arrays)
        var xs = List[Int]()
        var i = 0
        while i < len(samples):
            xs.append(samples[i])
            i += 1

        var j = 1
        while j < len(xs):
            var key = xs[j]
            var k = j - 1
            while k >= 0 and xs[k] > key:
                xs[k+1] = xs[k]
                k = k - 1
            xs[k+1] = key
            j += 1

        var n = len(xs)
        if n == 0:
            # Degenerate empty
            self.inner.add_series_json(
                String("{ 'name':") + _json_str(name) +
                String(", 'type':'line', 'data':[[0,0]], 'showSymbol':false, 'lineStyle':{ 'width':") + String(line_width) +
                String(" }, 'itemStyle':{ 'color':") + _json_str(color) + String(" }, 'step':") + _json_str(step_style) + String(" }")
            )
            return

        var ONE_M = 1000000
        var data_json = String("[")
        var first = True
        var idx = 0
        var cum = 0

        # We build step points at each unique x (right-continuous by default: 'end')
        while idx < n:
            var x = xs[idx]
            # count duplicates at this x
            var run = 1
            while (idx + run) < n and xs[idx + run] == x:
                run += 1
            cum += run  # cumulative count up to and including x

            # Build y either as count or normalized probability (0..1)
            var y_str = String("")
            if normalized:
                # y_scaled = floor(cum * 1e6 / n)
                var y_scaled = (cum * ONE_M) // n
                if survival:
                    # S = 1 - F  → s_scaled = 1e6 - y_scaled (clamp to [0,1e6])
                    var s_scaled = ONE_M - y_scaled
                    if s_scaled < 0: s_scaled = 0
                    if s_scaled > ONE_M: s_scaled = ONE_M
                    y_str = _format_scaled_6(s_scaled)
                else:
                    y_str = _format_scaled_6(y_scaled)
            else:
                # counts
                if survival:
                    var s_count = n - cum
                    y_str = String(s_count)
                else:
                    y_str = String(cum)

            if not first:
                data_json += String(",")
            data_json += String("[") + String(x) + String(",") + y_str + String("]")

            first = False
            idx += run

        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) + String(", 'type':'line', 'data':") + data_json
        obj += String(", 'step':") + _json_str(step_style)
        if show_symbol == False:
            obj += String(", 'showSymbol':false")
        obj += String(", 'lineStyle':{ 'width':") + String(line_width) + String(" }")
        obj += String(", 'itemStyle':{ 'color':") + _json_str(color) + String(" }")
        if area_opacity > 0:
            obj += String(", 'areaStyle':{ 'opacity':") + String(area_opacity) + String(" / 100 }")
        if show_labels:
            obj += String(", 'label':{ 'show':true, 'position':'top', 'formatter':") + _json_str(label_fmt) + String(" }")
        obj += String(" }")

        self.inner.add_series_json(obj)

 

    fn add_survival_series(mut self,
                        name: String,
                        samples: List[Int],
                        color: String,
                        normalized: Bool = True,
                        step_style: String = "end",
                        show_symbol: Bool = False,
                        line_width: Int = 2,
                        area_opacity: Int = 0,
                        show_labels: Bool = False,
                        label_fmt: String = "{c}"):
        self.add_ecdf_series(name, samples, color,
                                survival=True,
                                normalized=normalized,
                                step_style=step_style,
                                show_symbol=show_symbol,
                                line_width=line_width,
                                area_opacity=area_opacity,
                                show_labels=show_labels,
                                label_fmt=label_fmt)


    # Streamlines via ECharts 'lines' series on cartesian2d
    # - polylines: data = [ [[x0,y0],[x1,y1],...], [[...], ...], ... ]
    # - color/width/opacity control aesthetics; effect adds moving particles along the lines.
    fn add_streamlines_series(mut self,
                              name: String,
                              polylines_x: List[List[Float64]],
                              polylines_y: List[List[Float64]],
                              color: String = "#5470C6",
                              width: Int = 1,
                              opacity: Int = 85,
                              show_effect: Bool = True,
                              effect_speed: Int = 20,
                              effect_size: Int = 2,
                              trail_length: Int = 30):
        # Build data = [{ coords: [[x,y], ...] }, ...]
        var data_json = String("[")
        var i = 0
        while i < len(polylines_x) and i < len(polylines_y):
            var xs = polylines_x[i]
            var ys = polylines_y[i]
            if len(xs) > 1 and len(xs) == len(ys):
                data_json += String("{ 'coords': [")
                var j = 0
                while j < len(xs):
                    data_json += String("[") + String(xs[j]) + String(",") + String(ys[j]) + String("]")
                    if j + 1 < len(xs): data_json += String(",")
                    j += 1
                data_json += String("] }")
                # comma between items
                if i + 1 < len(polylines_x) and i + 1 < len(polylines_y): data_json += String(",")
            i += 1
        data_json += String("]")

        var obj = String("{ 'name':") + _json_str(name) +
                  String(", 'type':'lines', 'coordinateSystem':'cartesian2d', 'polyline':true") +
                  String(", 'lineStyle':{ 'color':") + _json_str(color) +
                  String(", 'width':") + String(width) +
                  String(", 'opacity':") + String(opacity) + String(" / 100 }") +
                  String(", 'data':") + data_json

        if show_effect:
            obj += String(", 'effect':{ 'show':true, 'constantSpeed':") + String(effect_speed) +
                   String(", 'trailLength':") + String(trail_length) + String(" / 100") +
                   String(", 'symbolSize':") + String(effect_size) + String(" }")

        obj += String(" }")
        self.inner.add_series_json(obj)
