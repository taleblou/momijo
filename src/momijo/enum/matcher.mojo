# Project:      Momijo
# Module:       src.momijo.enum.matcher
# File:         matcher.mojo
# Path:         src/momijo/enum/matcher.mojo
#
# Description:  src.momijo.enum.matcher — focused Momijo functionality with a stable public API.
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
#   - Structs: Matcher
#   - Key functions: __init__, __copyinit__, __moveinit__, build_matcher, match_with_selector
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.error import module
from momijo.dataframe.helpers import m
from momijo.ir.dialects.annotations import tags
from pathlib import Path
from pathlib.path import Path

struct Matcher:
    var tags: List[Int]
    var results: List[Int]
fn __init__(out self, tags: List[Int], results: List[Int]) -> None:
        self.tags = tags
        self.results = results
fn __copyinit__(out self, other: Self) -> None:
        self.tags = other.tags
        self.results = other.results
fn __moveinit__(out self, deinit other: Self) -> None:
        self.tags = other.tags
        self.results = other.results
fn build_matcher(tags: List[Int], results: List[Int]) -> Matcher:
    return Matcher(tags=tags, results=results)
fn match_with_selector(selector: Int, default: Int, m: Matcher) -> Int:
    for i in range(len(m.tags)):
        if m.tags[i] == selector:
            return m.results[i]
    return default