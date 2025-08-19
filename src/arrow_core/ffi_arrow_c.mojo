# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

# minimal opaque handles for Arrow C Data Interface.
from sys.ffi import UnsafePointer

struct CArrowArrayHandle:
    ptr: UnsafePointer[UInt8]
    fn __init__(inout self, ptr: UnsafePointer[UInt8]):
        self.ptr = ptr
    fn is_null(self) -> Bool:
        return self.ptr.is_null()

struct CArrowSchemaHandle:
    ptr: UnsafePointer[UInt8]
    fn __init__(inout self, ptr: UnsafePointer[UInt8]):
        self.ptr = ptr
    fn is_null(self) -> Bool:
        return self.ptr.is_null()

# wrap a raw address as array handle. Inputs: addr (address). Output: array handle
fn import_array_from_address(addr: UInt64) -> CArrowArrayHandle:
    var p = UnsafePointer[UInt8](addr)
    return CArrowArrayHandle(p)

# wrap a raw address as schema handle. Inputs: addr (address). Output: schema handle
fn import_schema_from_address(addr: UInt64) -> CArrowSchemaHandle:
    var p = UnsafePointer[UInt8](addr)
    return CArrowSchemaHandle(p)
