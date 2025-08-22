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
# File: momijo/arrow_core/ffi_arrow_c.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

struct ArrowArray(Copyable, Movable):
    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: Pointer[Pointer[Void]]
    var children: Pointer[Pointer[ArrowArray]]
    var dictionary: Pointer[ArrowArray]
    var release: Pointer[Void]
    var private_data: Pointer[Void]

struct ArrowSchema(Copyable, Movable):
    var format: Pointer[UInt8]     # C string
    var name: Pointer[UInt8]       # C string
    var metadata: Pointer[UInt8]   # C string
    var flags: Int64
    var n_children: Int64
    var children: Pointer[Pointer[ArrowSchema]]
    var dictionary: Pointer[ArrowSchema]
    var release: Pointer[Void]
    var private_data: Pointer[Void]

# Constants for ArrowSchema flags (from Arrow C Data Interface)
const ARROW_FLAG_DICTIONARY_ORDERED: Int64 = 1
const ARROW_FLAG_NULLABLE: Int64 = 2
const ARROW_FLAG_MAP_KEYS_SORTED: Int64 = 4

# Free functions for simple inspection

fn arrow_array_length(arr: ArrowArray) -> Int64:
    return arr.length

fn arrow_array_null_count(arr: ArrowArray) -> Int64:
    return arr.null_count

fn arrow_schema_flag_nullable(schema: ArrowSchema) -> Bool:
    return (schema.flags & ARROW_FLAG_NULLABLE) != 0
