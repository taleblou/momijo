# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/tables.mojo
# Description: Standard JPEG Huffman and Quantization tables (baseline).

# Returns 16 counts (code lengths 1..16) for DC
fn std_dc_lengths() -> UnsafePointer[UInt8]:
    # ITU T.81, Annex K.3, Table K.3 (DC Luminance/Chrominance share same lengths commonly)
    var vals = [ 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ]
    var p = UnsafePointer[UInt8].alloc(16)
    var i = 0
    while i < 16:
        p[i] = UInt8(vals[i])
        i = i + 1
    return p

# Returns DC symbols for the above lengths (luminance default)
fn std_dc_symbols() -> UnsafePointer[UInt8]:
    # ITU T.81, Annex K.3, Table K.3 (values 0..11)
    var vals = [ 0,1,2,3,4,5,6,7,8,9,10,11 ]
    var p = UnsafePointer[UInt8].alloc(vals.__len__())
    var i = 0
    while i < vals.__len__():
        p[i] = UInt8(vals[i])
        i = i + 1
    return p

# Returns 16 counts for AC
fn std_ac_lengths() -> UnsafePointer[UInt8]:
    # ITU T.81, Annex K.3, Table K.5 (AC Luminance/Chrominance lengths often identical here)
    var vals = [ 0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125 ]
    var p = UnsafePointer[UInt8].alloc(16)
    var i = 0
    while i < 16:
        p[i] = UInt8(vals[i])
        i = i + 1
    return p

# Returns AC symbols table (162 bytes per spec table K.5)
fn std_ac_symbols() -> UnsafePointer[UInt8]:
    # ITU T.81, Annex K.3, Table K.5 (standard luminance AC)
    var vals = [
        1,2,3,0,4,11,5,12,21,31,41,6,13,22,51,61,7,32,71,14,42,81,91,8,23,52,101,
        111,121,15,62,131,24,72,82,9,33,53,92,10,43,63,73,16,34,54,83,17,25,44,55,
        18,35,45,64,19,26,36,46,65,74,84,20,27,37,56,93,102,112,122,28,38,47,57,66,
        103,113,29,39,48,67,75,85,94,104,114,30,40,49,58,68,76,86,95,105,115,123,
        132,133,141,142,150,151,160,169,170,179,187,188,197,198,199,208,209,210,
        218,219,220,227,228,229,230,231,232,233,234,241,242,243,244,245,246,247,
        248,249,250
    ]
    var p = UnsafePointer[UInt8].alloc(vals.__len__())
    var i = 0
    while i < vals.__len__():
        p[i] = UInt8(vals[i])
        i = i + 1
    return p

# Standard Luma quantization (quality ~50)
fn std_luma_qt() -> UnsafePointer[UInt8]:
    var vals = [
        16,11,10,16,24,40,51,61,
        12,12,14,19,26,58,60,55,
        14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62,
        18,22,37,56,68,109,103,77,
        24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101,
        72,92,95,98,112,100,103,99
    ]
    var p = UnsafePointer[UInt8].alloc(64)
    var i = 0
    while i < 64:
        p[i] = UInt8(vals[i])
        i = i + 1
    return p

# Standard Chroma quantization (simple table)
fn std_chroma_qt() -> UnsafePointer[UInt8]:
    var vals = [
        17,18,18,24,21,24,47,26,
        26,47,99,66,56,66,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99
    ]
    var p = UnsafePointer[UInt8].alloc(64)
    var i = 0
    while i < 64:
        p[i] = UInt8(vals[i])
        i = i + 1
    return p
