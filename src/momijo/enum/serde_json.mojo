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
# File: momijo/enum/serde_json.mojo


from momijo.enum import EnumValue

fn enum_to_json(v: EnumValue) -> String:
    var out = String("{\"tag\":") + String(v.tag) + String(",\"words\":[") + String(v.w0) + String(",") + String(v.w1) + String(",") + String(v.w2) + String(",") + String(v.w3) + String("]}")
    return out

# Does: utility function in enum module.
# Inputs: s.
# Returns: result value or status.
fn enum_from_json(s: String) -> (Bool, EnumValue):
    var v = EnumValue(tag=0, w0=0, w1=0, w2=0, w3=0)
    var idx = 0
    while idx < len(s) and s[idx] != ':': idx += 1
    if idx >= len(s): return (False, v)
    idx += 1
    var tag: UInt64 = 0
    while idx < len(s) and s[idx] >= '0' and s[idx] <= '9':
        tag = tag * 10 + UInt64(Int(s[idx]) - Int('0'))
        idx += 1
    v.tag = tag
    while idx < len(s) and s[idx] != '[': idx += 1
    if idx >= len(s): return (False, v)
    idx += 1
    var arr = [0,0,0,0]
    var ai = 0
    var cur: UInt64 = 0; var in_num = False
    while idx < len(s) and ai < 4:
        var ch = s[idx]
        if ch >= '0' and ch <= '9':
            cur = cur * 10 + UInt64(Int(ch) - Int('0')); in_num = True
        else:
            if in_num:
                arr[ai] = cur; ai += 1; cur = 0; in_num = False
        idx += 1
    if in_num and ai < 4:
        arr[ai] = cur; ai += 1
    v.w0 = UInt64(arr[0]); v.w1 = UInt64(arr[1]); v.w2 = UInt64(arr[2]); v.w3 = UInt64(arr[3])
    return (True, v)

# Does: utility function in enum module.
# Inputs: v, nwords.
# Returns: result value or status.
fn enum_to_json_compact(v: EnumValue, nwords: Int) -> String:
    var out = String("{\"tag\":") + String(v.tag) + String(",\"words\":[")
    for i in range(0, nwords):
        var w = v.w0 if i==0 else (v.w1 if i==1 else (v.w2 if i==2 else v.w3))
        out = out + (String(",") if i>0 else String("")) + String(w)
    out = out + String("]}")
    return out

# Does: utility function in enum module.
# Inputs: xs.
# Returns: result value or status.
fn enum_list_to_json(xs: List[EnumValue]) -> String:
    var out = String("[")
    for i in range(0, len(xs)):
        out = out + (String(",") if i>0 else String("")) + enum_to_json(xs[i])
    out = out + String("]")
    return out