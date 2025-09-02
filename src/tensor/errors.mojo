# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: momijo/tensor/errors.mojo

struct ErrorKind(Copyable, Movable, EqualityComparable):
    var code: Int

    fn __init__(out self, code: Int):
        self.code = code

    fn __copyinit__(out self, other: Self):
        self.code = other.code

    @always_inline
    fn __eq__(self, other: ErrorKind) -> Bool:
        return self.code == other.code

    @always_inline
    fn __ne__(self, other: ErrorKind) -> Bool:
        return self.code != other.code

    fn to_string(self) -> String:
        var s = String("Unknown")
        if self.code == 0:   s = String("OK")
        elif self.code == 1: s = String("InvalidArgument")
        elif self.code == 2: s = String("ShapeMismatch")
        elif self.code == 3: s = String("DTypeMismatch")
        elif self.code == 4: s = String("DeviceUnavailable")
        elif self.code == 5: s = String("NotContiguous")
        elif self.code == 6: s = String("OutOfBounds")
        elif self.code == 7: s = String("MemoryAllocation")
        elif self.code == 8: s = String("ArithmeticOverflow")
        elif self.code == 9: s = String("DivideByZero")
        elif self.code == 10: s = String("NotImplemented")
        return s

    @staticmethod
    fn OK() -> ErrorKind:                 return ErrorKind(0)
    @staticmethod
    fn InvalidArgument() -> ErrorKind:    return ErrorKind(1)
    @staticmethod
    fn ShapeMismatch() -> ErrorKind:      return ErrorKind(2)
    @staticmethod
    fn DTypeMismatch() -> ErrorKind:      return ErrorKind(3)
    @staticmethod
    fn DeviceUnavailable() -> ErrorKind:  return ErrorKind(4)
    @staticmethod
    fn NotContiguous() -> ErrorKind:      return ErrorKind(5)
    @staticmethod
    fn OutOfBounds() -> ErrorKind:        return ErrorKind(6)
    @staticmethod
    fn MemoryAllocation() -> ErrorKind:   return ErrorKind(7)
    @staticmethod
    fn ArithmeticOverflow() -> ErrorKind: return ErrorKind(8)
    @staticmethod
    fn DivideByZero() -> ErrorKind:       return ErrorKind(9)
    @staticmethod
    fn NotImplemented() -> ErrorKind:     return ErrorKind(10)
    @staticmethod
    fn Unknown() -> ErrorKind:            return ErrorKind(11)


# ---------- Error object ----------
struct TensorError(Copyable, Movable):
    var kind: ErrorKind
    var message: String
    var where_: String

    fn __init__(out self, kind: ErrorKind, message: String = String(""), where_: String = String("")):
        self.kind = kind
        self.message = message
        self.where_ = where_

    fn __copyinit__(out self, other: TensorError):
        self.kind = other.kind
        self.message = other.message
        self.where_ = other.where_

    fn ok(self) -> Bool:
        return self.kind == ErrorKind.OK()

    fn to_string(self) -> String:
        var s = String("[") + self.kind.to_string() + String("]")
        if len(self.message) > 0:
            s = s + String(" ") + self.message
        if len(self.where_) > 0:
            s = s + String(" (at ") + self.where_ + String(")")
        return s

    @always_inline
    fn __eq__(self, other: TensorError) -> Bool:
        return (self.kind == other.kind) and (self.message == other.message) and (self.where_ == other.where_)

    @always_inline
    fn __ne__(self, other: TensorError) -> Bool:
        return not (self == other)


# ---------- Status (void result) ----------
struct Status(Copyable, Movable):
    var ok: Bool
    var err: TensorError

    fn __init__(out self, ok: Bool, err: TensorError):
        self.ok = ok
        self.err = err

    fn __copyinit__(out self, other: Status):
        self.ok = other.ok
        self.err = other.err

    fn is_ok(self) -> Bool:
        return self.ok

    fn is_err(self) -> Bool:
        return not self.ok

    fn error(self) -> TensorError:
        return self.err

    fn to_string(self) -> String:
        if self.ok:
            return String("OK")
        return self.err.to_string()

# Constructors
fn OkStatus() -> Status:
    return Status(True, TensorError(ErrorKind.OK(), String(""), String("")))

fn ErrStatus(kind: ErrorKind, message: String = String(""), where_: String = String("")) -> Status:
    return Status(False, TensorError(kind, message, where_))


# ---------- Result[T] (value result) ----------
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

    fn is_ok(self) -> Bool:
        return self.ok

    fn is_err(self) -> Bool:
        return not self.ok

    fn unwrap_or(self, default_value: T) -> T:
        if self.ok:
            return self.value
        return default_value

    fn error(self) -> TensorError:
        return self.err

# Constructors
fn Ok[T: Copyable & Movable](value: T) -> Result[T]:
    return Result[T](True, value, TensorError(ErrorKind.OK(), String(""), String("")))

fn Err[T: Copyable & Movable](kind: ErrorKind, message: String = String(""), where_: String = String("")) -> Result[T]:
    var dummy: T = __zeroed_value_for_T[T]()
    return Result[T](False, dummy, TensorError(kind, message, where_))

fn __zeroed_value_for_T[T: Copyable & Movable]() -> T:
    var tmp: T
    return tmp


# ---------- Convenience builders ----------
fn err_invalid_arg(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.InvalidArgument(), msg, where_)

fn err_shape_mismatch(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.ShapeMismatch(), msg, where_)

fn err_dtype_mismatch(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.DTypeMismatch(), msg, where_)

fn err_not_contiguous(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.NotContiguous(), msg, where_)

fn err_out_of_bounds(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.OutOfBounds(), msg, where_)

fn err_not_implemented(msg: String, where_: String = String("")) -> TensorError:
    return TensorError(ErrorKind.NotImplemented(), msg, where_)


# ---------- Aggregation ----------
struct ErrorList(Copyable, Movable, Sized):
    var items: List[TensorError]

    fn __init__(out self):
        self.items = List[TensorError]()

    fn __copyinit__(out self, other: ErrorList):
        self.items = other.items

    @always_inline
    fn __len__(self) -> Int:
        return len(self.items)

    fn push(mut self, e: TensorError) -> None:
        self.items.append(e)

    fn any(self) -> Bool:
        return len(self.items) > 0

    fn clear(mut self) -> None:
        self.items = List[TensorError]()

    fn join_messages(self) -> String:
        var out = String("")
        var first = True
        for e in self.items:
            if first:
                out = e.to_string()
                first = False
            else:
                out = out + String("; ") + e.to_string()
        return out


# ---------- Fail helper ----------
fn fail(msg: String) -> None:
    # Print the error message; do not terminate (keeps compilation/runtime simple).
    print(String("ERROR: ") + msg)
