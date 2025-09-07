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
# Project: momijo.core
# File: src/momijo/core/string_utils.mojo

from momijo.core.error import module
from momijo.dataframe.helpers import m
from pathlib import Path
from pathlib.path import Path
from sys import implementation

fn starts_with(s: String, prefix: String) -> Bool:
    # Naive implementation; replace with stdlib when available.
    var n = len(prefix)
    if len(s) < n:
        return False
    var i = 0
    while i < n:
        if s[i] != prefix[i]:
            return False
        i += 1
    return True
fn ends_with(s: String, suffix: String) -> Bool:
    var n = len(suffix)
    var m = len(s)
    if m < n:
        return False
    var i = 0
    while i < n:
        if s[m - n + i] != suffix[i]:
            return False
        i += 1
    return True