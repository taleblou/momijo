# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/export.mojo

from momijo.visual.types import Figure
from momijo.visual.features.registry import get_exporter

fn export_figure_html(fig: Figure, path: String, flavor: String = "echarts", auto_open: Bool = False) -> Bool:
    var opt = get_exporter(flavor)
    if opt is None:
        print("exporter not available: " + flavor)
        return False
    var ok = opt.value().export_html(fig, path)
    if ok and auto_open:
        from momijo.visual.platform import open_in_browser
        open_in_browser(path)
    return ok