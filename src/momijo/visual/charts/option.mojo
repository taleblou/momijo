# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/charts/option.mojo
# Description: Comprehensive ECharts Option builder with param-by-param API (2D-friendly).
# Notes:
#  - Stores JSON snippets as String fields and arrays of String.
#  - No private _set_kv is used; 3D keys (xAxis3D, yAxis3D, zAxis3D, grid3D) are injected by EasyOption.
#  - Compatible with demos (bar/line/scatter/pie) and with EasyOption-based 3D.

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

fn _join_with_comma(xs: List[String]) -> String:
    var out = String("")
    var i = 0
    while i < len(xs):
        out += xs[i]
        if i < len(xs) - 1:
            out += String(", ")
        i += 1
    return out

fn _json_array_str(xs: List[String]) -> String:
    var out = String("[")
    var i = 0
    while i < len(xs):
        out += String("'") + xs[i] + String("'")
        if i < len(xs) - 1:
            out += String(",")
        i += 1
    out += String("]")
    return out

fn _json_array_int(xs: List[Int]) -> String:
    var out = String("[")
    var i = 0
    while i < len(xs):
        out += String(xs[i])
        if i < len(xs) - 1:
            out += String(",")
        i += 1
    out += String("]")
    return out

# ------------------------------------------------------------------------------
# Simple series builders (optional sugar)
# ------------------------------------------------------------------------------

fn series_bar(name: String, data: List[Int]) -> String:
    var out = String("{ 'name':'") + name +
              String("', 'type':'bar', 'data':") + _json_array_int(data) +
              String(" }")
    return out

fn series_line(name: String, data: List[Int]) -> String:
    var out = String("{ 'name':'") + name +
              String("', 'type':'line', 'data':") + _json_array_int(data) +
              String(" }")
    return out

# ------------------------------------------------------------------------------
# Option: param-by-param builder
# ------------------------------------------------------------------------------

struct Option:
    # 1. Titles
    var title: String                 # JSON object

    # 2. Tooltip
    var tooltip: String               # JSON object

    # 3. Legend
    var legend: String                # JSON object

    # Polar (optional)
    var angleAxis: String             # JSON object
    var radiusAxis: String            # JSON object
    var polar: String                 # JSON object

    # 4. Grid
    var grid: String                  # JSON object

    # 5. Axes & coordinate systems
    var xAxis: String                 # JSON object (or array JSON if you pass it so)
    var yAxis: String                 # JSON object
    var singleAxis: String            # JSON object
    var radar: String                 # JSON object

    # 6. Series
    var series: List[String]          # array of JSON strings (each a series)

    # 7. Colors
    var color: List[String]           # ['#5470C6', '#91CC75', ...]

    # 8. Background
    var backgroundColor: String       # e.g. '#fff'

    # 9. Toolbox
    var toolbox: String               # JSON object

    # 10. DataZoom
    var dataZoom: List[String]        # array of JSON objects (as strings)

    # 11. VisualMap
    var visualMap: String             # JSON object

    # 12. Timeline
    var timeline: String              # JSON object

    # 13. Calendar
    var calendar: String              # JSON object

    # 14. Graphic
    var graphic: String               # JSON object

    # 15. Aria
    var aria: String                  # JSON object

    # 16. Animation
    var animation: Bool
    var animationDuration: Int
    var animationEasing: String

    # 17. AxisPointer
    var axisPointer: String           # JSON object

    # 18. Dataset
    var dataset: String               # JSON object

    # 19. Geo
    var geo: String                   # JSON object

    # 20. Parallel
    var parallel: String              # JSON object
    var parallelAxis: String          # JSON array or object

    # 3D axes & grid (optional)
    var xAxis3D: String
    var yAxis3D: String
    var zAxis3D: String
    var grid3D: String

    fn __init__(out self):
        self.title = String("")
        self.tooltip = String("")
        self.legend = String("")
        self.angleAxis = String("")
        self.radiusAxis = String("")
        self.polar = String("")
        self.grid = String("")
        self.xAxis = String("")
        self.yAxis = String("")
        self.singleAxis = String("")
        self.radar = String("")
        self.series = List[String]()
        self.color = List[String]()
        self.backgroundColor = String("")
        self.toolbox = String("")
        self.dataZoom = List[String]()
        self.visualMap = String("")
        self.timeline = String("")
        self.calendar = String("")
        self.graphic = String("")
        self.aria = String("")
        self.animation = True
        self.animationDuration = 1000
        self.animationEasing = String("cubicOut")
        self.axisPointer = String("")
        self.dataset = String("")
        self.geo = String("")
        self.parallel = String("")
        self.parallelAxis = String("")
        self.xAxis3D = String("")
        self.yAxis3D = String("")
        self.zAxis3D = String("")
        self.grid3D  = String("")


    # ---------------------------
    # Setters (objects as JSON)
    # ---------------------------

    fn set_title(mut self, title_json: String):
        self.title = title_json

    fn set_subtitle(mut self, text: String, subtext: String):
        self.title = String("{ 'text':'") + text +
                    String("', 'subtext':'") + subtext + String("' }")

    fn set_tooltip(mut self, tooltip_json: String):
        self.tooltip = tooltip_json

    fn set_legend(mut self, legend_json: String):
        self.legend = legend_json

    fn set_grid(mut self, grid_json: String):
        self.grid = grid_json

    fn set_background(mut self, color: String):
        self.backgroundColor = color

    fn set_toolbox(mut self, toolbox_json: String):
        self.toolbox = toolbox_json

    fn set_visual_map(mut self, visual_map_json: String):
        self.visualMap = visual_map_json

    fn set_timeline(mut self, timeline_json: String):
        self.timeline = timeline_json

    fn set_calendar(mut self, calendar_json: String):
        self.calendar = calendar_json

    fn set_graphic(mut self, graphic_json: String):
        self.graphic = graphic_json

    fn set_aria(mut self, aria_json: String):
        self.aria = aria_json

    fn set_axis_pointer(mut self, axis_pointer_json: String):
        self.axisPointer = axis_pointer_json

    fn set_dataset(mut self, dataset_json: String):
        self.dataset = dataset_json

    fn set_geo(mut self, geo_json: String):
        self.geo = geo_json

    fn set_parallel(mut self, parallel_json: String):
        self.parallel = parallel_json

    fn set_parallel_axis(mut self, json_obj: String):
        # Accept either full object containing 'parallelAxis' or just the array/object
        var s = json_obj.strip()
        if len(s) > 0 and s[0:1] == String("{") and (s.find(String("'parallelAxis'")) >= 0 or s.find(String("\"parallelAxis\"")) >= 0):
            var lb = s.find(String("["))
            var rb = s.rfind(String("]"))
            if lb >= 0 and rb > lb:
                self.parallelAxis = s[lb:rb+1]   # only the array part
            else:
                self.parallelAxis = json_obj
        else:
            self.parallelAxis = json_obj

    # Axes & coordinate systems
    fn set_xaxis_json(mut self, json_obj: String):
        self.xAxis = json_obj

    fn set_yaxis_json(mut self, json_obj: String):
        self.yAxis = json_obj

    fn set_single_axis(mut self, json_obj: String):
        self.singleAxis = json_obj

    fn set_polar(mut self, json_obj: String):
        self.polar = json_obj

    fn set_radar(mut self, json_obj: String):
        self.radar = json_obj

    # Polar quick helpers
    fn set_angle_axis(mut self, json_obj: String):
        self.angleAxis = json_obj

    fn set_radius_axis(mut self, json_obj: String):
        self.radiusAxis = json_obj

    # Quick helpers for common axes
    fn set_xaxis_category(mut self, categories: List[String]):
        self.xAxis = String("{ 'type':'category', 'data': ") +
                     _json_array_str(categories) + String(" }")

    fn set_yaxis_value(mut self):
        self.yAxis = String("{ 'type':'value' }")

    # Colors & series & dataZoom
    fn set_colors(mut self, colors: List[String]):
        self.color = colors

    fn add_color(mut self, color: String):
        self.color.append(color)

    fn add_series_json(mut self, s_json: String):
        self.series.append(s_json)

    fn add_datazoom_json(mut self, dz_json: String):
        self.dataZoom.append(dz_json)

    # Animation params
    fn set_animation(mut self, enabled: Bool, duration: Int = 1000, easing: String = String("cubicOut")):
        self.animation = enabled
        self.animationDuration = duration
        self.animationEasing = easing

    # ---------------------------
    # Finalization
    # ---------------------------

    fn to_json(self) -> String:
        var parts = List[String]()

        # Polar fallbacks: if 'polar' is set but angle/radius axes are empty, provide minimal defaults
        var angleAxis_local = self.angleAxis
        var radiusAxis_local = self.radiusAxis
        if len(self.polar) > 0:
            if len(angleAxis_local) == 0:
                angleAxis_local = String("{ 'type':'value', 'startAngle': 90 }")
            if len(radiusAxis_local) == 0:
                radiusAxis_local = String("{ }")

        # Collect set fields
        if len(self.title) > 0: parts.append(String("'title': ") + self.title)
        if len(self.tooltip) > 0: parts.append(String("'tooltip': ") + self.tooltip)
        if len(self.legend) > 0: parts.append(String("'legend': ") + self.legend)
        if len(self.grid) > 0: parts.append(String("'grid': ") + self.grid)

        if len(self.xAxis) > 0: parts.append(String("'xAxis': ") + self.xAxis)
        if len(self.yAxis) > 0: parts.append(String("'yAxis': ") + self.yAxis)
        if len(self.singleAxis) > 0: parts.append(String("'singleAxis': ") + self.singleAxis)
        if len(self.polar) > 0: parts.append(String("'polar': ") + self.polar)
        if len(angleAxis_local) > 0: parts.append(String("'angleAxis': ") + angleAxis_local)
        if len(radiusAxis_local) > 0: parts.append(String("'radiusAxis': ") + radiusAxis_local)
        if len(self.radar) > 0: parts.append(String("'radar': ") + self.radar)

        if len(self.series) > 0: parts.append(String("'series': [") + _join_with_comma(self.series) + String("]"))
        if len(self.dataZoom) > 0: parts.append(String("'dataZoom': [") + _join_with_comma(self.dataZoom) + String("]"))
        if len(self.visualMap) > 0: parts.append(String("'visualMap': ") + self.visualMap)
        if len(self.toolbox) > 0: parts.append(String("'toolbox': ") + self.toolbox)
        if len(self.backgroundColor) > 0: parts.append(String("'backgroundColor': '") + self.backgroundColor + String("'"))
        if len(self.color) > 0: parts.append(String("'color': ") + _json_array_str(self.color))

        if len(self.timeline) > 0: parts.append(String("'timeline': ") + self.timeline)
        if len(self.calendar) > 0: parts.append(String("'calendar': ") + self.calendar)
        if len(self.graphic) > 0: parts.append(String("'graphic': ") + self.graphic)
        if len(self.aria) > 0: parts.append(String("'aria': ") + self.aria)

        # Animation section (always included)
        parts.append(String("'animation': ") + (String("true") if self.animation else String("false")))
        parts.append(String("'animationDuration': ") + String(self.animationDuration))
        parts.append(String("'animationEasing': '") + self.animationEasing + String("'"))

        if len(self.axisPointer) > 0: parts.append(String("'axisPointer': ") + self.axisPointer)
        if len(self.dataset) > 0: parts.append(String("'dataset': ") + self.dataset)
        if len(self.geo) > 0: parts.append(String("'geo': ") + self.geo)
        if len(self.parallel) > 0: parts.append(String("'parallel': ") + self.parallel)
        if len(self.parallelAxis) > 0: parts.append(String("'parallelAxis': ") + self.parallelAxis)

        # Emit final JSON object
        var out = String("{ ")
        var k = 0
        while k < len(parts):
            out += parts[k]
            if k < len(parts) - 1:
                out += String(", ")
            k += 1
        out += String(" }")
        return out
    

 

    # Simple passthroughs in Option (if missing)
    fn set_xaxis3d_json(mut self, json: String):
        self.xAxis3D = json

    fn set_yaxis3d_json(mut self, json: String):
        self.yAxis3D = json

    fn set_zaxis3d_json(mut self, json: String):
        self.zAxis3D = json

    fn set_grid3d_json(mut self, json: String):
        self.grid3D = json
