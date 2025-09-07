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
# Project: momijo.enum
# File: momijo/enum/matcher.mojo


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