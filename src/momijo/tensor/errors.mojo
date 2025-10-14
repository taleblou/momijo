# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.errors
# File:         src/momijo/tensor/errors.mojo
#
# Description:
#   Error/Status/Result primitives (assert-free) for Momijo Tensor stack.
#   - ErrorKind: compact error codes with string conversion
#   - TensorError: structured error with kind/message/where
#   - Status: OK/ERR without payload
#   - Result[T]: payload + error
#   - Helpers: Ok/Err constructors, validation/check utilities, ErrorList

from collections.list import List

# ---------- ErrorKind ----------
struct ErrorKind(Copyable, Movable, EqualityComparable):
    var code: Int32

    @always_inline
    fn __eq__(self, other: ErrorKind) -> Bool: return self.code == other.code
    @always_inline
    fn __ne__(self, other: ErrorKind) -> Bool: return self.code != other.code

    @staticmethod @always_inline fn OK()                -> ErrorKind: return ErrorKind(Int32(0))
    @staticmethod @always_inline fn InvalidArgument()  -> ErrorKind: return ErrorKind(Int32(1))
    @staticmethod @always_inline fn ShapeMismatch()    -> ErrorKind: return ErrorKind(Int32(2))
    @staticmethod @always_inline fn DTypeMismatch()    -> ErrorKind: return ErrorKind(Int32(3))
    @staticmethod @always_inline fn DeviceUnavailable()-> ErrorKind: return ErrorKind(Int32(4))
    @staticmethod @always_inline fn NotContiguous()    -> ErrorKind: return ErrorKind(Int32(5))
    @staticmethod @always_inline fn OutOfBounds()      -> ErrorKind: return ErrorKind(Int32(6))
    @staticmethod @always_inline fn MemoryAllocation() -> ErrorKind: return ErrorKind(Int32(7))
    @staticmethod @always_inline fn ArithmeticOverflow()-> ErrorKind: return ErrorKind(Int32(8))
    @staticmethod @always_inline fn DivideByZero()     -> ErrorKind: return ErrorKind(Int32(9))
    @staticmethod @always_inline fn NotImplemented()   -> ErrorKind: return ErrorKind(Int32(10))

    fn ravel(self) -> String:
        var t = self.code
        if t == 0:  return String("OK")
        if t == 1:  return String("InvalidArgument")
        if t == 2:  return String("ShapeMismatch")
        if t == 3:  return String("DTypeMismatch")
        if t == 4:  return String("DeviceUnavailable")
        if t == 5:  return String("NotContiguous")
        if t == 6:  return String("OutOfBounds")
        if t == 7:  return String("MemoryAllocation")
        if t == 8:  return String("ArithmeticOverflow")
        if t == 9:  return String("DivideByZero")
        return String("NotImplemented")

    @always_inline
    fn __str__(self) -> String: return self.ravel()

# ---------- TensorError ----------
struct TensorError(Copyable, Movable):
    var kind: ErrorKind
    var message: String
    var where_: String

    @always_inline
    fn ok(self) -> Bool:
        return self.kind == ErrorKind.OK()

    fn ravel(self) -> String:
        var s = String("[") + self.kind.ravel() + String("]")
        if len(self.message) > 0:
            s = s + String(" ") + self.message
        if len(self.where_) > 0:
            s = s + String(" (at ") + self.where_ + String(")")
        return s

    @always_inline
    fn __str__(self) -> String: return self.ravel()

# ---------- Status ----------
struct Status(Copyable, Movable):
    var ok: Bool
    var err: TensorError

    fn __init__(out self, ok: Bool, err: TensorError):
        self.ok = ok
        self.err = err

    @always_inline fn is_ok(self) -> Bool: return self.ok
    @always_inline fn is_err(self) -> Bool: return not self.ok
    @always_inline fn error(self) -> TensorError: return self.err

    fn ravel(self) -> String:
        if self.ok: return String("OK")
        return self.err.ravel()

    @always_inline
    fn __str__(self) -> String: return self.ravel()

@always_inline fn OkStatus() -> Status:
    return Status(True, TensorError(ErrorKind.OK(), String(""), String("")))

@always_inline fn ErrStatus(kind: ErrorKind, message: String = String(""), where_: String = String("")) -> Status:
    return Status(False, TensorError(kind, message, where_))

# ---------- Result[T] ----------
struct Result[T: Copyable & Movable](Copyable, Movable):
    var ok: Bool
    var value: T
    var err: TensorError

    fn __init__(out self, ok: Bool, value: T, err: TensorError):
        self.ok = ok
        self.value = value
        self.err = err

    fn __copyinit__(out self, other: Result[T]):
        self.ok = other.ok
        self.value = other.value
        self.err = other.err

    @always_inline fn is_ok(self) -> Bool: return self.ok
    @always_inline fn is_err(self) -> Bool: return not self.ok

    @always_inline
    fn unwrap_or(self, default_value: T) -> T:
        if self.ok: return self.value
        return default_value

    @always_inline fn error(self) -> TensorError: return self.err

@always_inline
fn Ok[T: Copyable & Movable](value: T) -> Result[T]:
    return Result[T](True, value, TensorError(ErrorKind.OK(), String(""), String("")))

@always_inline
fn Err[T: Copyable & Movable](kind: ErrorKind, message: String = String(""), where_: String = String("")) -> Result[T]:
    var dummy: T = __zeroed_value_for_T[T]()
    return Result[T](False, dummy, TensorError(kind, message, where_))

@always_inline
fn __zeroed_value_for_T[T: Copyable & Movable]() -> T:
    var tmp: T
    return tmp

# ---------- Fast helpers ----------
@always_inline fn err_invalid_arg(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.InvalidArgument(), msg, where_)
@always_inline fn err_shape_mismatch(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.ShapeMismatch(), msg, where_)
@always_inline fn err_dtype_mismatch(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.DTypeMismatch(), msg, where_)
@always_inline fn err_not_contiguous(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.NotContiguous(), msg, where_)
@always_inline fn err_out_of_bounds(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.OutOfBounds(), msg, where_)
@always_inline fn err_not_implemented(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.NotImplemented(), msg, where_)

# ---------- Logging (no assert, no throw) ----------
@always_inline
fn fail(msg: String) -> None:
    print(String("ERROR: ") + msg)

# ---------- Require/Ensure (assert-free) ----------
@always_inline
fn require(cond: Bool, msg: String) -> Bool:
    if not cond:
        fail(msg)
        return False
    return True

# ---------- ErrorList ----------
struct ErrorList(Copyable, Movable, Sized):
    var items: List[TensorError]

    fn __init__(out self):
        self.items = List[TensorError]()

    @always_inline fn push(mut self, e: TensorError) -> None:
        self.items.append(e)

    @always_inline fn any(self) -> Bool:
        return len(self.items) > 0

    @always_inline fn clear(mut self) -> None:
        self.items = List[TensorError]()

    @always_inline
    fn __len__(self) -> Int:
        return len(self.items)

    fn join_messages(self) -> String:
        var out = String("")
        var first = True
        for e in self.items:
            if first:
                out = e.ravel()
                first = False
            else:
                out = out + String("; ") + e.ravel()
        return out

# ---------- Checks (assert-free) ----------
@always_inline
fn check_axis(axis: Int, ndim: Int, ctx: String = String("axis")) -> Bool:
    if ndim < 0:
        fail(String("invalid ndim < 0")); return False
    if axis < 0 or axis >= ndim:
        fail(ctx + String(" out of range: ") + String(axis) + String(" vs ndim ") + String(ndim))
        return False
    return True

@always_inline
fn check_equal_len(a: Int, b: Int, what: String) -> Bool:
    if a != b:
        fail(what + String(" length mismatch: ") + String(a) + String(" vs ") + String(b))
        return False
    return True

fn check_same_shape(a: List[Int], b: List[Int], what: String = String("shape")) -> Bool:
    if len(a) != len(b):
        fail(what + String(" rank mismatch: ") + String(len(a)) + String(" vs ") + String(len(b)))
        return False
    var i = 0
    var n = len(a)
    while i < n:
        if a[i] != b[i]:
            fail(
                what + String(" mismatch at dim ")
                + String(i) + String(": ") + String(a[i]) + String(" vs ") + String(b[i])
            )
            return False
        i += 1
    return True

fn check_product(shape: List[Int], expected: Int, what: String = String("shape product")) -> Bool:
    var size = 1
    var i = 0
    var n = len(shape)
    while i < n:
        size = size * shape[i]
        i += 1
    if size != expected:
        fail(what + String(" mismatch: ") + String(size) + String(" vs ") + String(expected))
        return False
    return True

@always_inline
fn check_non_empty(n: Int, what: String = String("length")) -> Bool:
    if n <= 0:
        fail(what + String(" must be > 0"))
        return False
    return True

@always_inline
fn check_positive(x: Int, what: String) -> Bool:
    if x <= 0:
        fail(what + String(" must be > 0, got ") + String(x))
        return False
    return True

fn check_broadcastable_like(a: List[Int], b: List[Int], ctx: String = String("broadcast")) -> Bool:
    # Numpy-like right-aligned broadcasting compatibility check.
    var ia = len(a) - 1
    var ib = len(b) - 1
    while ia >= 0 or ib >= 0:
        var da = 1
        var db = 1
        if ia >= 0: da = a[ia]
        if ib >= 0: db = b[ib]
        var ok = (da == db) or (da == 1) or (db == 1)
        if not ok:
            fail(
                ctx + String(" incompatible at dims (")
                + String(ia) + String(",") + String(ib) + String("): ")
                + String(da) + String(" vs ") + String(db)
            )
            return False
        ia -= 1
        ib -= 1
    return True

@always_inline
fn check_axis_status(axis: Int, ndim: Int, ctx: String = String("axis")) -> Status:
    if check_axis(axis, ndim, ctx): return OkStatus()
    return ErrStatus(ErrorKind.InvalidArgument(), ctx, String("check_axis"))

@always_inline
fn check_same_shape_status(a: List[Int], b: List[Int], what: String = String("shape")) -> Status:
    if check_same_shape(a, b, what): return OkStatus()
    return ErrStatus(ErrorKind.ShapeMismatch(), what, String("check_same_shape"))

@always_inline
fn check_broadcastable_status(a: List[Int], b: List[Int], ctx: String = String("broadcast")) -> Status:
    if check_broadcastable_like(a, b, ctx): return OkStatus()
    return ErrStatus(ErrorKind.InvalidArgument(), ctx, String("check_broadcastable_like"))
