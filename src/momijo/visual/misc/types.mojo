# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/types.mojo

struct Figure:
    var _title: String
    var _y_name: String
    var _w: Int
    var _h: Int
    var _x_labels: List[String]
    var _y_series: List[List[Float64]]
    var _series_names: List[String]

    fn __init__(out self,
                title: String,
                y_name: String,
                width_px: Int,
                height_px: Int,
                x_labels: List[String],
                y_series: List[List[Float64]],
                series_names: List[String]):
        self._title = title
        self._y_name = y_name
        self._w = width_px
        self._h = height_px
        self._x_labels = x_labels
        self._y_series = y_series
        self._series_names = series_names

    fn title(self) -> String: return self._title
    fn y_label(self) -> String: return self._y_name
    fn width_px(self) -> Int: return self._w
    fn height_px(self) -> Int: return self._h

    fn to_line_triplet(self) -> (List[String], List[List[Float64]], List[String]):
        return (self._x_labels, self._y_series, self._series_names)