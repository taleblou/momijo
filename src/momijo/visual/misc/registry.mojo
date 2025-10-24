# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/features/registry.mojo

from momijo.visual.types import Figure

trait ChartExporter:
    fn name(self) -> String
    fn export_html(self, fig: Figure, path: String) -> Bool

fn get_exporter(flavor: String) -> Optional[ChartExporter]:
    if flavor == String("echarts"):
        from momijo.visual.features.echarts_exporter import make_echarts_exporter
        return make_echarts_exporter()
    return None