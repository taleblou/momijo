# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/echarts_axes.mojo
# Description: Axes

from momijo.visual.echarts_core import json_array_str_quoted as _json_array_str_quoted
fn axis_value() -> String: return "{\"type\":\"value\"}"
fn axis_log(base: Int = 10) -> String: return "{\"type\":\"log\",\"logBase\":" + String(base) + "}"
fn axis_time() -> String: return "{\"type\":\"time\"}"
fn axis_category(labels: List[String]) -> String: return "{\"type\":\"category\",\"data\":" + _json_array_str_quoted(labels) + "}"
fn angle_axis_category(labels: List[String]) -> String: return "{\"type\":\"category\",\"data\":" + _json_array_str_quoted(labels) + "}"
fn radius_axis_value() -> String: return "{\"type\":\"value\"}"
fn parallel_axis(dim: Int, name: String, axis_type: String = "value") -> String: return "{\"dim\":" + String(dim) + ",\"name\":\"" + name + "\",\"type\":\"" + axis_type + "\"}"