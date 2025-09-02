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
# Project: momijo.core
# File: momijo/core/parameter.mojo

from builtin.dtype import DType
from momijo.core.error import MomijoError
from momijo.core.result import Result
from logger import Logger, Level
from momijo.core.tensor import Tensor

# -------------------------
# ParameterState (for serialization-like uses)
# -------------------------

@fieldwise_init
struct ParameterState(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

    var key: String
    var value: Tensor
    var grad: Tensor
    var has_grad: Bool
    var requires_grad: Bool
    var dtype: DType
    var device: String

    fn __init__(out self self,
        key: String,
        value: Tensor,
        grad: Tensor,
        has_grad: Bool,
        requires_grad: Bool,
        dtype: DType,
        device: String):
        self.key = key
        self.value() = value
        self.grad = grad
        self.has_grad = has_grad
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.device = device

    fn to_string(self) -> String:
        var g = self.has_grad ? "grad=present" : "grad=absent"
        return "ParameterState{key=" + self.key + ", " + g + ", dtype=" + self.dtype.to_string() + ", device=" + self.device + "}"


# -------------------------
# Parameter
# -------------------------

@fieldwise_init
struct Parameter(Copyable, Movable):
    var tensor: Tensor
    var grad: Tensor             # not guaranteed to be initialized; see has_grad
    var name: String
    var requires_grad: Bool
    var has_grad: Bool
    var dtype: DType
    var device: String

    fn __init__(out self self,
        tensor: Tensor,
        name: String = "",
        requires_grad: Bool = True,
        dtype: DType = DType.f32(),
        device: String = "cpu",
        has_grad: Bool = False,
        grad: Tensor = tensor          # placeholder; do not rely on grad unless has_grad == true
    ):
        self.tensor = tensor
        self.grad = grad
        self.name = name
        self.requires_grad = requires_grad
        self.has_grad = has_grad
        self.dtype = dtype
        self.device = device

    fn __copyinit__(out self, other: Self):
        self.tensor = other.tensor
        self.grad = other.grad
        self.name = other.name
        self.requires_grad = other.requires_grad
        self.has_grad = other.has_grad
        self.dtype = other.dtype
        self.device = other.device

    # ------- metadata -------
    fn with_name(self, name: String) -> Parameter:
        var p = self
        p.name = name
        return p

    fn with_requires_grad(self, enabled: Bool) -> Parameter:
        var p = self
        p.requires_grad = enabled
        return p

    fn with_dtype(self, dtype: DType) -> Parameter:
        var p = self
        p.dtype = dtype
        return p

    fn with_device(self, device: String) -> Parameter:
        var p = self
        p.device = device
        return p

    # ------- grad management -------
    fn attach_grad(self, grad: Tensor) -> Parameter:
        var p = self
        p.grad = grad
        p.has_grad = True
        return p

    fn clear_grad(self) -> Parameter:
        var p = self
        p.has_grad = False
        return p

    # ------- safe update hooks (no math assumptions) -------
    fn apply_inplace(self, f: fn(Tensor) -> Tensor) -> None:
        # Let caller define how to transform the tensor. We simply assign.
        self.tensor = f(self.tensor)

    fn apply_grad_inplace(self, f: fn(Tensor, Tensor) -> Tensor) -> None:
        # Applies f(param, grad) -> new_param if grad is present; otherwise no-op.
        if self.has_grad and self.requires_grad:
            self.tensor = f(self.tensor, self.grad)

    # ------- state io -------
    fn state(self, key: String) -> ParameterState:
        return ParameterState(
            key=key,
            value=self.tensor,
            grad=self.grad,
            has_grad=self.has_grad,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            device=self.device
        )

    @staticmethod
    fn from_state(s: ParameterState) -> Parameter:
        return Parameter(
            tensor=s.value(),
            name=s.key,
            requires_grad=s.requires_grad,
            dtype=s.dtype,
            device=s.device,
            has_grad=s.has_grad,
            grad=s.grad
        )

    # ------- validation -------
    fn validate(self) -> Result[Parameter]:
        # Without assuming tensor APIs, we only validate name and flags.
        if self.name.len() == 0:
            return Result[Parameter].fail(MomijoError.invalid_argument("parameter name is empty", "momijo.core.parameter"), self)
        return Result[Parameter].ok(self)

    # ------- presentation -------
    fn to_string(self) -> String:
        var g = self.has_grad ? "has_grad" : "no_grad"
        var r = self.requires_grad ? "requires_grad" : "frozen"
        var nm = (self.name.len() > 0) ? self.name : "<unnamed>"
        return "Parameter{" + nm + ", " + r + ", " + g + ", dtype=" + self.dtype.to_string() + ", device=" + self.device + "}"


# -------------------------
# Parameter group (optimizer-friendly metadata)
# -------------------------

@fieldwise_init
struct ParamHyper(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

    var lr: Float64
    var weight_decay: Float64
    var momentum: Float64

    fn __init__(out self self, lr: Float64 = 1e-3, weight_decay: Float64 = 0.0, momentum: Float64 = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum


@fieldwise_init
struct ParameterGroup(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

    var name: String
    var params: List[Parameter]
    var hyper: ParamHyper

    fn __init__(out self self, name: String = "", params: List[Parameter] = List[Parameter](), hyper: ParamHyper = ParamHyper()):
        self.name = name
        self.params = params
        self.hyper = hyper

    fn add(self, p: Parameter) -> None:
        self.params.append(p)

    fn size(self) -> Int:
        return len(self.params)

    fn to_string(self) -> String:
        return "ParameterGroup{name=" + (self.name.len() > 0 ? self.name : "<group>") + ", size=" + String(self.size()) + "}"


# -------------------------
# Utilities
# -------------------------

@staticmethod
fn param(tensor: Tensor, name: String = "", requires_grad: Bool = True, dtype: DType = DType.f32(), device: String = "cpu") -> Parameter:
    return Parameter(tensor=tensor, name=name, requires_grad=requires_grad, dtype=dtype, device=device, has_grad=False, grad=tensor)

@staticmethod
fn tie_weight(p: Parameter, q: Parameter) -> (Parameter, Parameter):
    # Share the same tensor reference (shallow tie). Grad flags kept as-is.
    var pp = p
    var qq = q
    qq.tensor = pp.tensor
    return (pp, qq)