# Project:      Momijo
# Module:       learn.engine.jit_compile
# File:         engine/jit_compile.mojo
# Path:         src/momijo/learn/engine/jit_compile.mojo
#
# Description:  JIT/Graph compilation facade for Momijo Learn. Provides a stable, backend-agnostic
#               API to switch between eager-like execution and a future graph-compiled mode.
#               The default implementation is a no-op that preserves the public surface.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Types: JITOptions, JITReport
#   - Key fns: compile(model, graph), compile_with(model, opts), dry_trace(fn, ...)
#   - Backend-agnostic stubs; wire real tracer/graph lowering later.
#   - No globals; callers should hold onto options/reports as needed.


# -----------------------------------------------------------------------------
# Options & Report
# -----------------------------------------------------------------------------

struct JITOptions:
    var graph: Bool
    var optimize: Bool
    var check_numerics: Bool
    var allow_mixed_precision: Bool

    fn __init__(
        out self,
        graph: Bool = False,
        optimize: Bool = True,
        check_numerics: Bool = False,
        allow_mixed_precision: Bool = True
    ):
        self.graph = graph
        self.optimize = optimize
        self.check_numerics = check_numerics
        self.allow_mixed_precision = allow_mixed_precision


struct JITReport:
    var graph_enabled: Bool
    var passes_applied: Int
    var lowered_ops: Int
    var fallback_ops: Int
    var notes: String

    fn __init__(
        out self,
        graph_enabled: Bool = False,
        passes_applied: Int = 0,
        lowered_ops: Int = 0,
        fallback_ops: Int = 0,
        notes: String = String("")
    ):
        self.graph_enabled = graph_enabled
        self.passes_applied = passes_applied
        self.lowered_ops = lowered_ops
        self.fallback_ops = fallback_ops
        self.notes = notes

    fn __str__(self) -> String:
        # Minimal string builder; extend when Writable is in place
        var s = String("JITReport(")
        s += String("graph=") + String(self.graph_enabled)
        s += String(", passes=") + String(self.passes_applied)
        s += String(", lowered_ops=") + String(self.lowered_ops)
        s += String(", fallback_ops=") + String(self.fallback_ops)
        s += String(", notes='") + self.notes + String("')")
        s += String(")")
        return s


# -----------------------------------------------------------------------------
# Public Facade
# -----------------------------------------------------------------------------
# Keep the original signature to remain source-compatible with existing calls.
# Currently returns the model unchanged; internally, we treat this as a hint
# for the training engine to prefer a graph execution path when available.

fn compile(model, graph: Bool = False):
    # NOTE: This default implementation does not mutate/replace the model.
    # A future backend can:
    #  - record a trace on first invocation,
    #  - build a static graph,
    #  - attach an executor handle to `model`,
    #  - or return a proxy that delegates to compiled code.
    _ = graph  # unused in the stub; prevents "unused variable" warnings
    return model


# A more explicit variant that accepts extended options and yields a JITReport.
# Returns the (possibly) same model for now, plus a report for logging/debug.
fn compile_with(model, opts: JITOptions, out report: JITReport):
    # Placeholder: no transformation performed yet
    report = JITReport(
        graph_enabled=opts.graph,
        passes_applied=Int(0),
        lowered_ops=Int(0),
        fallback_ops=Int(0),
        notes=String("No-op compile (backend not wired).")
    )
    return model


# -----------------------------------------------------------------------------
# Tracing Utilities (dry run)
# -----------------------------------------------------------------------------
# These helpers give you a place to wire a future tracer. For now, they just
# return synthetic info so the rest of the stack can be developed/tested.

# Simulate a lightweight trace to inspect callability and I/O arity.
# Returns a minimal "IR string" you can print or log during development.
fn dry_trace(fn_ref, inputs_arity: Int = 1) -> String:
    var ir = String("; momijo.learn.engine.jit_compile.dry_trace\n")
    ir += String("; fn_ref=") + String("<callable>") + String("\n")
    ir += String("; inputs_arity=") + String(inputs_arity) + String("\n")
    ir += String("; NOTE: tracer not implemented; this is a stub.\n")
    return ir


# -----------------------------------------------------------------------------
# Execution Policy (hinting)
# -----------------------------------------------------------------------------
# Some callers (e.g., Trainer) might want a simple boolean to branch behavior.

fn prefer_graph(opts: JITOptions) -> Bool:
    # Later you can add heuristics: tensor shapes, static control flow, etc.
    if opts.graph:
        return True
    return False


# -----------------------------------------------------------------------------
# Example of a minimal "executor" shim (no-op)
# -----------------------------------------------------------------------------
# This shows how a future compiled executor might be represented and used by
# the engine, without forcing a specific tensor/model API today.

struct GraphExecutor:
    var is_valid: Bool

    fn __init__(out self, is_valid: Bool = False):
        self.is_valid = is_valid

    fn run(self, model, batch):
        # In a real backend, this would dispatch the compiled function.
        # For now, just forward to the model's eager path (duck-typed).
        # If model has a `forward` or `__call__` convention, call it here.
        # Placeholder: return the batch unchanged to keep type-agnostic.
        return batch


# Utility to "build" an executor from options; currently a stub.
fn build_executor(opts: JITOptions) -> GraphExecutor:
    if opts.graph:
        return GraphExecutor(True)
    return GraphExecutor(False)
