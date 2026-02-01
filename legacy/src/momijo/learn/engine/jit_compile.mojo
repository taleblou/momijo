# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.engine.jit_compile
# File:         src/momijo/learn/engine/jit_compile.mojo
#
# Description:
#   JIT/Graph compilation facade for Momijo Learn. Provides a stable,
#   backend-agnostic API to switch between eager-like execution and a future
#   graph-compiled path. The default implementation is a no-op that preserves
#   the public surface and allows the rest of the stack (trainer, loops) to
#   develop against stable types and functions.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: JITOptions, JITReport, GraphExecutor
#   - Key fns: compile(model, graph), compile_with(model, opts, out report),
#              dry_trace(fn_ref, inputs_arity), prefer_graph(opts),
#              build_executor(opts)
#   - Optimizer/model/tensor APIs are intentionally not imported here.
#   - No globals; no wildcard imports; var-only style.

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

    # Fluent helpers to derive tweaked copies of options
    fn with_graph(self, on: Bool) -> JITOptions:
        var o = self
        o.graph = on
        return o

    fn with_optimize(self, on: Bool) -> JITOptions:
        var o = self
        o.optimize = on
        return o

    fn with_check_numerics(self, on: Bool) -> JITOptions:
        var o = self
        o.check_numerics = on
        return o

    fn with_mixed_precision(self, on: Bool) -> JITOptions:
        var o = self
        o.allow_mixed_precision = on
        return o

    fn __str__(self) -> String:
        var s = String("JITOptions(")
        s = s + "graph=" + (String("true") if self.graph else String("false"))
        s = s + ", optimize=" + (String("true") if self.optimize else String("false"))
        s = s + ", check_numerics=" + (String("true") if self.check_numerics else String("false"))
        s = s + ", allow_mixed_precision=" + (String("true") if self.allow_mixed_precision else String("false"))
        s = s + ")"
        return s


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
        var s = String("JITReport(")
        s = s + "graph=" + (String("true") if self.graph_enabled else String("false"))
        s = s + ", passes=" + String(self.passes_applied)
        s = s + ", lowered_ops=" + String(self.lowered_ops)
        s = s + ", fallback_ops=" + String(self.fallback_ops)
        s = s + ", notes='" + self.notes + "')"
        return s


# -----------------------------------------------------------------------------
# Public Facade
# -----------------------------------------------------------------------------

# Keep the original signature for source compatibility. Currently a no-op.
fn compile(model, graph: Bool = False):
    _ = graph  # intentionally unused in stub
    return model

# A more explicit variant that accepts options and yields a report (via out).
# Returns the (possibly the same) model for now.
fn compile_with(model, opts: JITOptions, out report: JITReport):
    report = JITReport(
        graph_enabled=opts.graph,
        passes_applied=Int(0),
        lowered_ops=Int(0),
        fallback_ops=Int(0),
        notes=String("No-op compile: backend not wired.")
    )
    return model


# -----------------------------------------------------------------------------
# Tracing Utilities (dry run)
# -----------------------------------------------------------------------------

# Simulate a lightweight trace to inspect callability/arity in development.
fn dry_trace(fn_ref, inputs_arity: Int = 1) -> String:
    var ir = String("; momijo.learn.engine.jit_compile.dry_trace\n")
    ir = ir + "; fn_ref=" + "<callable>" + "\n"
    ir = ir + "; inputs_arity=" + String(inputs_arity) + "\n"
    ir = ir + "; NOTE: tracer not implemented; this is a stub.\n"
    return ir


# -----------------------------------------------------------------------------
# Execution Policy (hinting)
# -----------------------------------------------------------------------------

# Simple policy that callers (e.g., Trainer) can query to branch behavior.
fn prefer_graph(opts: JITOptions) -> Bool:
    if opts.graph:
        return True
    return False


# -----------------------------------------------------------------------------
# Minimal "executor" shim (no-op)
# -----------------------------------------------------------------------------

struct GraphExecutor:
    var is_valid: Bool

    fn __init__(out self, is_valid: Bool = False):
        self.is_valid = is_valid

    # In a real backend, this would run the compiled graph. Here, just forward.
    fn run(self, model, batch):
        # Duck-typed pattern: if model has forward(batch), prefer it; else return batch.
        # We avoid reflective calls here to stay language-safe; a higher-level wrapper
        # can decide how to delegate. For stub purposes, return batch unchanged.
        _ = model
        return batch

    fn __str__(self) -> String:
        var s = String("GraphExecutor(valid=")
        s = s + (String("true") if self.is_valid else String("false")) + ")"
        return s


# Factory for an executor based on options (stubbed).
fn build_executor(opts: JITOptions) -> GraphExecutor:
    if opts.graph:
        return GraphExecutor(True)
    return GraphExecutor(False)
