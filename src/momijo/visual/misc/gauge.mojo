# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts/gauge.mojo
# Description: Gauge chart builder.

from momijo.visual.echarts.core import json_string

fn echarts_gauge_option(value: Float64, title: String = "Gauge", unit: String = "%") -> String:
    var opt = String("{title:{text:")+json_string(title)+String("},tooltip:{},series:[{type:\"gauge\",detail:{formatter:\"{value}")+unit+String("\"},data:[{value:")+String(value)+String("}]}]}")
    return opt