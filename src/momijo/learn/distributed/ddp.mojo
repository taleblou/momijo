# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.distributed.ddp
# File:         src/momijo/learn/distributed/ddp.mojo
#
# Description:
#   Minimal DDP-like scaffolding (placeholders) for examples and early experiments.
#   NOTE: This does NOT implement real communication/all-reduce; it mirrors the
#   control flow and keeps a clean API surface for future backends.

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear  # kept concrete to avoid trait deps

# -----------------------------------------------------------------------------
# Process group (stub)
# -----------------------------------------------------------------------------
struct ProcessGroup(Copyable, Movable):
    var backend: String
    var rank: Int
    var world_size: Int

    fn __init__(out self, backend: String, rank: Int, world_size: Int):
        self.backend = backend
        self.rank = rank
        self.world_size = world_size

    fn __copyinit__(out self, other: Self):
        self.backend = other.backend
        self.rank = other.rank
        self.world_size = other.world_size

fn init_process_group(backend: String, rank: Int, world_size: Int) -> ProcessGroup:
    print("[ddp:init] backend:", backend, "| rank:", rank, "| world_size:", world_size)
    return ProcessGroup(backend, rank, world_size)

fn destroy_process_group(pg: ProcessGroup):
    print("[ddp:destroy] backend:", pg.backend, "| rank:", pg.rank)

# -----------------------------------------------------------------------------
# DDP wrapper (forward-only stub for Linear)
# -----------------------------------------------------------------------------
struct DDPLinear(Copyable, Movable):
    var module: Linear

    fn __init__(out self, module: Linear):
        # keep an owned copy to avoid aliasing surprises
        self.module = module.copy()

    fn __copyinit__(out self, other: Self):
        # value semantics
        self.module = other.module

    @staticmethod
    fn from_module(module: Linear) -> DDPLinear:
        return DDPLinear(module)

    fn replace_module(mut self, module: Linear) -> None:
        # allow swapping the wrapped module safely
        self.module = module.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # NOTE: only Float64 path provided to match current Linear API.
        # Add an overload if your Linear supports Float32 forwards too.
        return self.module.forward(x)
