# Project:      Momijo
# Module:       src.momijo.enum.unique
# File:         unique.mojo
# Path:         src/momijo/enum/unique.mojo
#
# Description:  src.momijo.enum.unique — focused Momijo functionality with a stable public API.
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
#   - Key functions: ensure_unique_tags, ensure_unique_names


from momijo.core.error import module
from momijo.dataframe.helpers import unique
from momijo.ir.dialects.annotations import tags
from pathlib import Path
from pathlib.path import Path

fn ensure_unique_tags(tags: List[UInt64]) -> Bool:
    var n = len(tags)
    for i in range(0, n):
        for j in range(i+1, n):
            if tags[i] == tags[j]:
                return False
    return True

# Does: utility function in enum module.
# Inputs: names.
# Returns: result value or status.
fn ensure_unique_names(names: List[String]) -> Bool:
    var n = len(names)
    for i in range(0, n):
        for j in range(i+1, n):
            if names[i] == names[j]:
                return False
    return True