# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

# minimal opaque handles for Arrow C Data Interface.
from sys.ffi import UnsafePointer

# Struct CArrowArrayHandle: auto-generated docs. Update as needed.
struct CArrowArrayHandle:
    ptr: UnsafePointer[UInt8]
# Constructor: __init__(out self, ptr: UnsafePointer[UInt8])
    fn __init__(out self, ptr: UnsafePointer[UInt8]):
        self.ptr = ptr
# Function is_null(self) -> Bool
    fn is_null(self) -> Bool:
        return self.ptr.is_null()

# Struct CArrowSchemaHandle: auto-generated docs. Update as needed.
struct CArrowSchemaHandle:
    ptr: UnsafePointer[UInt8]
# Constructor: __init__(out self, ptr: UnsafePointer[UInt8])
    fn __init__(out self, ptr: UnsafePointer[UInt8]):
        self.ptr = ptr
# Function is_null(self) -> Bool
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