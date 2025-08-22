# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/types.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 
 

from momijo.enum import Enum

struct ArrowType: Enum:
    INT
    FLOAT64
    STRING
    BOOL
    UNKNOWN

fn arrow_type_name(t: ArrowType) -> String:
    match t:
        case .INT: return "Int"
        case .FLOAT64: return "Float64"
        case .STRING: return "String"
        case .BOOL: return "Bool"
        case .UNKNOWN: return "Unknown"

# Map from string name to ArrowType
fn parse_arrow_type(name: String) -> ArrowType:
    if name == "Int":
        return ArrowType.INT
    if name == "Float64":
        return ArrowType.FLOAT64
    if name == "String":
        return ArrowType.STRING
    if name == "Bool":
        return ArrowType.BOOL
    return ArrowType.UNKNOWN

# Simple trait simulation: check if numeric
fn arrow_type_is_numeric(t: ArrowType) -> Bool:
    return t == ArrowType.INT or t == ArrowType.FLOAT64

# Default values for types
fn arrow_type_default(t: ArrowType) -> String:
    match t:
        case .INT: return "0"
        case .FLOAT64: return "0.0"
        case .STRING: return ""
        case .BOOL: return "false"
        case .UNKNOWN: return ""
