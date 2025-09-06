# ============================================================================
#  Momijo Visualization - runtime/theme.mojo
#  Copyright (c) 2025  Morteza Talebou  (https://taleblou.ir/)
#  Licensed under the MIT License. See LICENSE in the project root.
# ============================================================================

struct Theme:
    var name: String
    var background: String
    var axis_color: String
    var grid_color: String
    var text_color: String
    fn __init__(out self, name: String, background: String, axis_color: String, grid_color: String, text_color: String):
        self.name = name
        self.background = background
        self.axis_color = axis_color
        self.grid_color = grid_color
        self.text_color = text_color

@staticmethod
fn theme_scientific() -> Theme:
    return Theme(String("scientific"), String("#ffffff"), String("#222222"), String("#e5e5e5"), String("#111111"))

@staticmethod
fn theme_dark() -> Theme:
    return Theme(String("dark"), String("#111111"), String("#eeeeee"), String("#444444"), String("#f5f5f5"))

@staticmethod
fn theme_publisher() -> Theme:
    return Theme(String("publisher"), String("#ffffff"), String("#000000"), String("#d9d9d9"), String("#000000"))
