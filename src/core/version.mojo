# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/version.mojo

# No global variables. Provide stable, inlinable accessors.

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

# Lightweight self-test (does not run automatically; call manually if needed)
fn __self_test__():
    let s = version_string()
    assert(s == "0.1.0", "version_string mismatch: " + s)
