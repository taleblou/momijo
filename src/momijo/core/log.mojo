# Project:      Momijo
# Module:       src.momijo.core.log
# File:         log.mojo
# Path:         src/momijo/core/log.mojo
#
# Description:  src.momijo.core.log — focused Momijo functionality with a stable public API.
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
#   - Key functions: _prefix, info, warn, error


fn _prefix(level: String) -> String:
    return "[" + level + "] "
fn info(msg: String) -> None:
    print(_prefix("INFO") + msg)
fn warn(msg: String) -> None:
    print(_prefix("WARN") + msg)
fn error(msg: String) -> None:
    print(_prefix("ERROR") + msg)