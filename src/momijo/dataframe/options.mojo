# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dataframe
# File: src/momijo/dataframe/options.mojo

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