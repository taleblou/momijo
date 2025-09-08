# Project:      Momijo
# Module:       src.momijo.dataframe.options
# File:         options.mojo
# Path:         src/momijo/dataframe/options.mojo
#
# Description:  src.momijo.dataframe.options — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: Options
#   - Key functions: __init__, __copyinit__, with_options, __copyinit__


from momijo.core.option import __copyinit__
from momijo.dataframe.sampling import __init__
from momijo.extras.stubs import Self, display_rows, print

struct Options(Copyable, Movable):
    var display_rows: Int
fn __init__(out out self, outout self, display_rows: Int) -> None:
        self.display_rows = display_rows
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn with_options(mut opt: Options, action: String) -> None
    # no-op context (illustrative)
    print(String("with_options: display_rows=") + String(opt.display_rows) + String(" -> ") + action)
fn __copyinit__(out self, other: Self) -> None:

        self.display_rows = other.display_rows