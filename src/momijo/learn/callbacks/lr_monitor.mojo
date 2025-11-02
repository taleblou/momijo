# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.callbacks.lr_monitor
# File:         src/momijo/learn/callbacks/lr_monitor.mojo
#
# Description:
#   Learning-rate monitor callback for Momijo Learn.
#   - LRRecord: single snapshot of one or multiple LR values.
#   - LRMonitor: per-step / per-epoch logging, in-memory history, stdout printing.
#   Optimizer-agnostic and backend-agnostic: trainer passes LRs explicitly or
#   wires an optimizer adapter exposing current_lrs().
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# -----------------------------------------------------------------------------
# A single LR snapshot (can contain multiple parameter groups' LRs).
# -----------------------------------------------------------------------------
struct LRRecord:
    var step: Int
    var epoch: Int
    var lrs: List[Float64]

    fn __init__(out self, step: Int, epoch: Int, lrs: List[Float64]):
        self.step = step
        self.epoch = epoch
        self.lrs = lrs

    fn __str__(self) -> String:
        var s = String("LRRecord(step=") + String(self.step) + String(", epoch=") + String(self.epoch) + String(", lrs=[")
        var i = 0
        while i < len(self.lrs):
            s = s + String(self.lrs[i])
            if i + 1 < len(self.lrs):
                s = s + String(", ")
            i = i + 1
        s = s + String("])")
        return s


# -----------------------------------------------------------------------------
# Learning-rate monitor callback.
# -----------------------------------------------------------------------------
struct LRMonitor:
    var log_every_n_steps: Int
    var log_on_epoch_end: Bool
    var print_to_stdout: Bool

    var _history: List[LRRecord]
    var _last_logged_step: Int
    var _last_logged_epoch: Int

    fn __init__(
        out self,
        log_every_n_steps: Int = 1,
        log_on_epoch_end: Bool = True,
        print_to_stdout: Bool = True
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_on_epoch_end = log_on_epoch_end
        self.print_to_stdout = print_to_stdout

        self._history = List[LRRecord]()
        self._last_logged_step = -1
        self._last_logged_epoch = -1

    # -------------------------------------------------------------------------
    # Trainer hooks (to be called by your Trainer/Engine)
    # -------------------------------------------------------------------------
    fn on_train_start(mut self):
        # Reserved for warmup/header printing  
        pass

    fn on_batch_end(mut self, step: Int, epoch: Int, lrs: List[Float64]):
        if self.log_every_n_steps > 0:
            var should_log = (step % self.log_every_n_steps) == 0
            if should_log:
                self.record_step(step, epoch, lrs)

    fn on_epoch_end(mut self, epoch: Int, lrs: List[Float64]):
        if self.log_on_epoch_end:
            self.record_epoch(epoch, lrs)

    # -------------------------------------------------------------------------
    # Recording helpers (public)
    # -------------------------------------------------------------------------
    fn record_step(mut self, step: Int, epoch: Int, lrs: List[Float64]):
        var rec = LRRecord(step, epoch, lrs)
        self._history.append(rec)
        self._last_logged_step = step
        self._last_logged_epoch = epoch
        if self.print_to_stdout:
            var line = LRMonitor.format_line(String("step"), step, epoch, lrs)
            print(line)

    fn record_epoch(mut self, epoch: Int, lrs: List[Float64]):
        # Convention: step = -1 for epoch-level entries
        var rec = LRRecord(-1, epoch, lrs)
        self._history.append(rec)
        self._last_logged_epoch = epoch
        if self.print_to_stdout:
            var line = LRMonitor.format_line(String("epoch"), -1, epoch, lrs)
            print(line)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    fn last(self) -> LRRecord:
        if len(self._history) == 0:
            var empty = List[Float64]()
            return LRRecord(-1, -1, empty)
        return self._history[len(self._history) - 1]

    fn history_len(self) -> Int:
        return Int(len(self._history))

    fn history_at(self, i: Int) -> LRRecord:
        # Caller must if 0 <= i < history_len()
        return self._history[i]

    fn reset(mut self):
        self._history = List[LRRecord]()
        self._last_logged_step = -1
        self._last_logged_epoch = -1

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    fn format_line(kind: String, step: Int, epoch: Int, lrs: List[Float64]) -> String:
        # Example: "[LR][step=120, epoch=3] groups=2 [0.01, 0.001]"
        var prefix = String("[LR][") + kind + String("=")
        if kind == String("step"):
            prefix = prefix + String(step) + String(", epoch=") + String(epoch) + String("]")
        else:
            prefix = prefix + String(epoch) + String("]")

        var suffix = String(" groups=") + String(len(lrs.)) + String(" [")
        var i = 0
        while i < len(lrs):
            suffix = suffix + String(lrs[i])
            if i + 1 < len(lrs):
                suffix = suffix + String(", ")
            i = i + 1
        suffix = suffix + String("]")
        return prefix + suffix

    # Optional convenience for optimizer integration:
    # Requires optimizer.current_lrs() -> List[Float64]
    fn try_log_from_optimizer(mut self, step: Int, epoch: Int, optimizer, at_epoch_end: Bool = False):
        if at_epoch_end:
            var lrs_end = optimizer.current_lrs()
            self.record_epoch(epoch, lrs_end)
        else:
            var lrs_step = optimizer.current_lrs()
            self.record_step(step, epoch, lrs_step)

    # Optional: expose last logged indices (for debugging/telemetry)
    fn last_logged_step(self) -> Int:
        return self._last_logged_step

    fn last_logged_epoch(self) -> Int:
        return self._last_logged_epoch
