# Project:      Momijo
# Module:       src.momijo.core.version
# File:         version.mojo
# Path:         src/momijo/core/version.mojo
#
# Description:  src.momijo.core.version — focused Momijo functionality with a stable public API.
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
#   - Structs: Version
#   - Key functions: version_major, version_minor, version_patch, version_string, major, minor, patch, as_string ...
#   - Static methods present.


from momijo.autograd.hook import call
from momijo.core.error import module
from momijo.ir.llir.runtime_calls import assert
from pathlib import Path
from pathlib.path import Path
from sys import version

@always_inline
fn version_major() -> Int:
    return 0

@always_inline
fn version_minor() -> Int:
    return 1

@always_inline
fn version_patch() -> Int:
    return 0

@always_inline
fn version_string() -> String:
    return String(version_major()) + "." + String(version_minor()) + "." + String(version_patch())

# Optional: a compact tuple helper if you need it elsewhere.
struct Version:
    @staticmethod
fn major() -> Int: return version_major()
    @staticmethod
fn minor() -> Int: return version_minor()
    @staticmethod
fn patch() -> Int: return version_patch()
    @staticmethod
fn as_string() -> String: return version_string()
fn __init__(out self, ) -> None:
        pass
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# Lightweight self-test (does not run automatically; call manually if needed)
fn __self_test__() -> None:
    var s = version_string()
    assert(s == "0.1.0", "version_string mismatch: " + s)