# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT

from std.collections import Optional


@fieldwise_init
struct TensorStatusCode(Equatable, ImplicitlyCopyable):
    """Status for checked General tensor operations."""

    var code: Int

    comptime OK = Self(0)
    comptime INVALID_RANK = Self(1)  # Coordinate count differs from shape rank.
    comptime INVALID_SHAPE = Self(2)  # A dimension is negative.
    comptime OVERFLOW = Self(
        3
    )  # An element count or byte size exceeds Int.MAX.
    comptime OUT_OF_BOUNDS = Self(4)  # A coordinate is outside its axis.
    comptime SIZE_MISMATCH = Self(7)  # A reduction needs at least one element.
    comptime UNINITIALIZED = Self(27)  # The result's value has been taken.

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


struct TensorResult[T: Movable & Deinitable](Movable):
    """Holds a value or a failure status, including for move-only owners."""

    var _status: TensorStatusCode
    var _value: Optional[Self.T]

    def __init__(out self, var value: Self.T):
        self._status = TensorStatusCode.OK
        self._value = Optional[Self.T](value^)

    def __init__(out self, error: TensorStatusCode):
        self._status = error
        if error == TensorStatusCode.OK:
            self._status = TensorStatusCode.UNINITIALIZED
        self._value = None

    def status(self) -> TensorStatusCode:
        """Returns OK exactly while a value is available."""
        return self._status

    def is_ok(self) -> Bool:
        return self._status == TensorStatusCode.OK

    def take_value(mut self) -> Self.T:
        """Moves out the value. Calling this requires is_ok() to be true."""
        var value = self._value.take()
        self._status = TensorStatusCode.UNINITIALIZED
        return value^
