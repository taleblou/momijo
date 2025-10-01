# Project:      Momijo
# Module:       learn.callbacks.lr_monitor
# File:         callbacks/lr_monitor.mojo
# Path:         src/momijo/learn/callbacks/lr_monitor.mojo
#
# Description:  Learning-rate monitor callback for Momijo Learn. Collects and records
#               optimizer learning rates over training/validation, supports per-step
#               and per-epoch logging, and exposes a simple in-memory history API.
#               Backend-agnostic and optimizer-agnostic: the trainer may pass LRs
#               explicitly, or you can wire a tiny adapter on your optimizer to
#               expose current LR(s).
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
#   - Types: LRRecord, LRMonitor
#   - Key fns (trainer-facing):
#       * on_train_start()
#       * on_batch_end(step, epoch, lrs)
#       * on_epoch_end(epoch, lrs)
#       * record_step(step, epoch, lrs)
#       * record_epoch(epoch, lrs)
#   - Accessors:
#       * last() -> LRRecord
#       * history_len() -> Int
#       * history_at(i: Int) -> LRRecord
#       * reset()
#   - Optimizer-agnostic helpers:
#       * format_line(...): String
#       * try_log_from_optimizer(...) stub for future wiring

from collections.list import List

# A single LR snapshot (can contain multiple parameter groups' LRs).
struct LRRecord:
    var step: Int
    var epoch: Int
    var lrs: List[Float64]

    fn __init__(out self, step: Int, epoch: Int, lrs: List[Float64]):
        self.step = step
        self.epoch = epoch
        self.lrs = lrs

# Learning-rate monitor callback.
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
    # Trainer hook stubs (to be called by your Trainer/Engine)
    # -------------------------------------------------------------------------

    fn on_train_start(mut self):
        # No-op for now; reserved for future warmup, header printing, etc.
        pass

    fn on_batch_end(mut self, step: Int, epoch: Int, lrs: List[Float64]):
        # Record per-step when cadence matches log_every_n_steps
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
        self._history.push_back(rec)
        self._last_logged_step = step
        self._last_logged_epoch = epoch
        if self.print_to_stdout:
            var line = LRMonitor.format_line(String("step"), step, epoch, lrs)
            print(line)

    fn record_epoch(mut self, epoch: Int, lrs: List[Float64]):
        # Use step = -1 for epoch-level entries (convention)
        var rec = LRRecord(-1, epoch, lrs)
        self._history.push_back(rec)
        self._last_logged_epoch = epoch
        if self.print_to_stdout:
            var line = LRMonitor.format_line(String("epoch"), -1, epoch, lrs)
            print(line)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    fn last(self) -> LRRecord:
        if self._history.size() == 0:
            # Return an empty/default record (epoch/step = -1, no LRs)
            var empty = List[Float64]()
            return LRRecord(-1, -1, empty)
        return self._history[self._history.size() - 1]

    fn history_len(self) -> Int:
        return Int(self._history.size())

    fn history_at(self, i: Int) -> LRRecord:
        # Caller must ensure 0 <= i < history_len()
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
        # Example: "[LR][step=120, epoch=3] groups=2 [0.010000, 0.001000]"
        var prefix = String("[LR][") + kind + String("=")
        if kind == String("step"):
            prefix = prefix + String(step)
            prefix = prefix + String(", epoch=") + String(epoch) + String("]")
        else:
            prefix = prefix + String(epoch) + String("]")

        var suffix = String(" groups=") + String(Int(lrs.size())) + String(" [")
        var i = 0
        while i < lrs.size():
            suffix = suffix + String(lrs[i])
            if i + 1 < lrs.size():
                suffix = suffix + String(", ")
            i = i + 1
        suffix = suffix + String("]")
        return prefix + suffix

    # Optional convenience for future optimizer integration:
    # If your optimizer exposes a method `current_lrs() -> List[Float64]`, you can use this.
    fn try_log_from_optimizer(mut self, step: Int, epoch: Int, optimizer, at_epoch_end: Bool = False):
        if at_epoch_end:
            # Expect optimizer.current_lrs()
            var lrs = optimizer.current_lrs()
            self.record_epoch(epoch, lrs)
        else:
            var lrs = optimizer.current_lrs()
            self.record_step(step, epoch, lrs)
