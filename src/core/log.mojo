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
# File: momijo/core/log.mojo

# Very small logging helpers; printable-only; no external sinks.

fn _prefix(level: String) -> String:
    return "[" + level + "] "

fn info(msg: String):
    print(_prefix("INFO") + msg)

fn warn(msg: String):
    print(_prefix("WARN") + msg)

fn error(msg: String):
    print(_prefix("ERROR") + msg)
