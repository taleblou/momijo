# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/features/echarts_exporter.mojo

from momijo.visual.features.registry import ChartExporter
from momijo.visual.types import Figure
from momijo.visual.echarts.core import echarts_html, write_text
from momijo.visual.echarts.line import echarts_line_option

@parameter
struct EChartsConfig:
    var enabled: Bool = True

struct EChartsExporter:
    fn name(self) -> String:
        return String("echarts")

    fn export_html(self, fig: Figure, path: String) -> Bool:
        var (x_labels, y_data, series_names) = fig.to_line_triplet()
        var opt = echarts_line_option(x_labels, y_data, series_names, smooth=True, area=True,
                                      title=fig.title(), y_name=fig.y_label())
        var html = echarts_html(opt, title=fig.title(), width_px=fig.width_px(), height_px=fig.height_px())
        return write_text(path, html)

fn make_echarts_exporter() -> Optional[ChartExporter]:
    if EChartsConfig.enabled == False:
        return None
    var exp = EChartsExporter()
    return Optional[ChartExporter](exp)