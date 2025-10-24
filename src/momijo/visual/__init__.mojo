# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: /mnt/data/momijo_visual_final/src/momijo/visual/__init__.mojo
# Description: Aggregated public API for this package. English comments only.
from momijo.visual.core.core import echarts_html
from momijo.visual.core.core import json_array_str
from momijo.visual.core.core import json_number_array
from momijo.visual.core.core import json_string
from momijo.visual.core.core import json_string_array
from momijo.visual.core.core import write_text
from momijo.visual.core.echarts_core import json_array_f64
from momijo.visual.plotting.echarts_exporter import EChartsConfig
from momijo.visual.plotting.echarts_exporter import EChartsExporter
from momijo.visual.plotting.echarts_axes import angle_axis_category
from momijo.visual.plotting.echarts_axes import axis_category
from momijo.visual.plotting.echarts_axes import axis_log
from momijo.visual.plotting.echarts_axes import axis_time
from momijo.visual.plotting.echarts_axes import axis_value
from momijo.visual.plotting.echarts_series import chart_bar
from momijo.visual.plotting.echarts_3d import chart_bar3d
from momijo.visual.plotting.echarts_series import chart_boxplot
from momijo.visual.plotting.echarts_custom import chart_contour_lines
from momijo.visual.plotting.echarts_custom import chart_errorbar
from momijo.visual.plotting.echarts_custom import chart_eventplot
from momijo.visual.plotting.echarts_3d import chart_geo3d_world
from momijo.visual.plotting.echarts_3d import chart_globe_with_textures
from momijo.visual.plotting.echarts_series import chart_heatmap
from momijo.visual.plotting.echarts_custom import chart_hexbin
from momijo.visual.plotting.echarts_series import chart_line
from momijo.visual.plotting.echarts_3d import chart_line3d
from momijo.visual.plotting.echarts_3d import chart_lines3d
from momijo.visual.plotting.echarts_series import chart_pie
from momijo.visual.plotting.echarts_polar import chart_polar_bar
from momijo.visual.plotting.echarts_polar import chart_polar_line
from momijo.visual.plotting.echarts_polar import chart_polar_scatter
from momijo.visual.plotting.echarts_custom import chart_quiver
from momijo.visual.plotting.echarts_sankey import chart_sankey
from momijo.visual.plotting.echarts_series import chart_scatter
from momijo.visual.plotting.echarts_3d import chart_scatter3d
from momijo.visual.plotting.echarts_custom import chart_stem
from momijo.visual.plotting.echarts_3d import chart_surface3d
from momijo.visual.plotting.echarts_custom import chart_violin
from momijo.visual.plotting.echarts_layout import dataset_builder
from momijo.visual.plotting.boxplot import echarts_boxplot_option
from momijo.visual.plotting.graph import echarts_graph_option
from momijo.visual.plotting.echarts_exporter import export_html
from momijo.visual.plotting.echarts_exporter import make_echarts_exporter
from momijo.visual.plotting.echarts_exporter import name
from momijo.visual.plotting.echarts_axes import parallel_axis
from momijo.visual.plotting.echarts_axes import radius_axis_value
from momijo.visual.plotting.echarts_layout import timeline_builder
from momijo.visual.plotting.echarts_layout import visualMap_continuous
from momijo.visual.plotting.echarts_layout import visualMap_piecewise
from momijo.visual.html.html_viewer import render_echarts
from momijo.visual.misc.registry import ChartExporter
from momijo.visual.misc.types import Figure
from momijo.visual.misc.bar import echarts_bar_option
from momijo.visual.misc.calendar import echarts_calendar_option
from momijo.visual.misc.candlestick import echarts_candlestick_option
from momijo.visual.misc.custom import echarts_custom_option
from momijo.visual.misc.dashboard import echarts_dashboard_html
from momijo.visual.misc.effectscatter import echarts_effectscatter_option
from momijo.visual.misc.funnel import echarts_funnel_option
from momijo.visual.misc.gauge import echarts_gauge_option
from momijo.visual.misc.heatmap import echarts_heatmap_option
from momijo.visual.misc.line import echarts_line_option
from momijo.visual.misc.lines import echarts_lines_option
from momijo.visual.misc.map import echarts_map_option
from momijo.visual.misc.parallel import echarts_parallel_option
from momijo.visual.misc.pictorialbar import echarts_pictorialbar_option
from momijo.visual.misc.pie import echarts_pie_option
from momijo.visual.misc.radar import echarts_radar_option
from momijo.visual.misc.sankey import echarts_sankey_option
from momijo.visual.misc.scatter import echarts_scatter_option
from momijo.visual.misc.sunburst import echarts_sunburst_option
from momijo.visual.misc.themeriver import echarts_themeriver_option
from momijo.visual.misc.treemap import echarts_treemap_option
from momijo.visual.misc.export import export_figure_html
from momijo.visual.misc.registry import get_exporter
from momijo.visual.misc.types import height_px
from momijo.visual.misc.histogram import hist1d
from momijo.visual.misc.kde import kde_1d
from momijo.visual.misc.marching_squares import marching_squares
from momijo.visual.misc.themes import theme_dark
from momijo.visual.misc.themes import theme_light
from momijo.visual.misc.themes import theme_vintage
from momijo.visual.misc.types import title
from momijo.visual.misc.types import to_line_triplet
from momijo.visual.misc.types import width_px
from momijo.visual.misc.types import y_label
# Convenience factory for a minimal Figure (width/height only)
# Convenience factory for a minimal Figure with sane defaults.
fn new_figure(width_px: Int, height_px: Int, title: String = String(""), y_name: String = String("")) -> Figure:
    var xs = List[String]()
    var ys = List[List[Float64]]()
    var names = List[String]()
    return Figure(title, y_name, width_px, height_px, xs, ys, names)
from momijo.visual.html.preview import render_html
from momijo.visual.html.preview import open_in_browser
from momijo.visual.html.preview import write_preview_html


from momijo.visual.charts.option import Option
from momijo.visual.html.preview import render_html
from momijo.visual.charts.ez_option import EasyOption   
from momijo.visual.plotting.violin_utils import _compute_violin_curve_json, _ints   

 
fn preview(option_json: String,
           title: String = String("ECharts Demo"),
           width_px: Int = 960,
           height_px: Int = 750,
           out_path: String = String("chart.html"),
           auto_open: Bool = True,
           full_page: Bool = True,
           include_gl: Bool = False,
           pre_init_js: String = String(""),
           include_wordcloud_plugin: Bool = False) -> String:
    return render_html(option_json,
                       title=title,
                       width_px=width_px,
                       height_px=height_px,
                       include_world_map=False,
                       save=True,
                       auto_open=auto_open,
                       out_path=out_path,
                       full_page=full_page,
                       include_gl=include_gl,
                       pre_init_js=pre_init_js,
                       include_wordcloud_plugin=include_wordcloud_plugin)


