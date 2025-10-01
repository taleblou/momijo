# Project:      Momijo
# Module:       learn.engine.hooks
# File:         engine/hooks.mojo
# Path:         src/momijo/learn/engine/hooks.mojo
#
# Description:  Training event system (hooks) for Momijo Learn. Provides a
#               lightweight, backend-agnostic pub/sub mechanism to register
#               handlers for training lifecycle events (train/valid/test).
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
#   - Types: TrainState, BatchInfo, LRUpdateInfo, Hooks
#   - Handlers: function pointers with fixed signatures (no globals)
#   - Register via add_on_* methods; emit via on_* methods in the Trainer
#   - Safe defaults (no-op if no handlers); clear() to remove all handlers

from collections.list import List

# -----------------------------------------------------------------------------
# Shared state payloads passed to handlers
# -----------------------------------------------------------------------------

struct TrainState:
    var epoch: Int
    var global_step: Int
    var max_epochs: Int
    var steps_per_epoch: Int
    var train_loss: Float64
    var val_loss: Float64
    var lr: Float64

    fn __init__(
        out self,
        epoch: Int = 0,
        global_step: Int = 0,
        max_epochs: Int = 0,
        steps_per_epoch: Int = 0,
        train_loss: Float64 = 0.0,
        val_loss: Float64 = 0.0,
        lr: Float64 = 0.0
    ):
        self.epoch = epoch
        self.global_step = global_step
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.lr = lr


struct BatchInfo:
    var batch_index: Int
    var batch_size: Int

    fn __init__(out self, batch_index: Int = 0, batch_size: Int = 0):
        self.batch_index = batch_index
        self.batch_size = batch_size


struct LRUpdateInfo:
    var prev_lr: Float64
    var new_lr: Float64

    fn __init__(out self, prev_lr: Float64 = 0.0, new_lr: Float64 = 0.0):
        self.prev_lr = prev_lr
        self.new_lr = new_lr


# -----------------------------------------------------------------------------
# Hooks: an event bus with typed handler lists
# Handlers are function pointers; signatures are consistent per event type.
# -----------------------------------------------------------------------------

struct Hooks:
    # Train-level
    var _on_train_start: List[fn(mut TrainState)]
    var _on_train_end:   List[fn(mut TrainState)]

    # Epoch-level
    var _on_epoch_start: List[fn(mut TrainState)]
    var _on_epoch_end:   List[fn(mut TrainState)]

    # Batch-level (training)
    var _on_batch_start: List[fn(mut TrainState, BatchInfo)]
    var _on_batch_end:   List[fn(mut TrainState, BatchInfo, Float64)]  # loss

    # Validation/Test phases
    var _on_validation_start: List[fn(mut TrainState)]
    var _on_validation_end:   List[fn(mut TrainState)]
    var _on_test_start:       List[fn(mut TrainState)]
    var _on_test_end:         List[fn(mut TrainState)]

    # Scheduler / Checkpoint events
    var _on_lr_update:        List[fn(mut TrainState, LRUpdateInfo)]
    var _on_checkpoint_saved: List[fn(mut TrainState, String)]  # path

    fn __init__(out self):
        self._on_train_start = List[fn(mut TrainState)]()
        self._on_train_end   = List[fn(mut TrainState)]()

        self._on_epoch_start = List[fn(mut TrainState)]()
        self._on_epoch_end   = List[fn(mut TrainState)]()

        self._on_batch_start = List[fn(mut TrainState, BatchInfo)]()
        self._on_batch_end   = List[fn(mut TrainState, BatchInfo, Float64)]()

        self._on_validation_start = List[fn(mut TrainState)]()
        self._on_validation_end   = List[fn(mut TrainState)]()
        self._on_test_start       = List[fn(mut TrainState)]()
        self._on_test_end         = List[fn(mut TrainState)]()

        self._on_lr_update        = List[fn(mut TrainState, LRUpdateInfo)]()
        self._on_checkpoint_saved = List[fn(mut TrainState, String)]()

    # -----------------------------
    # Registration API (add_*)
    # -----------------------------

    fn add_on_train_start(mut self, h: fn(mut TrainState)):
        self._on_train_start.push_back(h)

    fn add_on_train_end(mut self, h: fn(mut TrainState)):
        self._on_train_end.push_back(h)

    fn add_on_epoch_start(mut self, h: fn(mut TrainState)):
        self._on_epoch_start.push_back(h)

    fn add_on_epoch_end(mut self, h: fn(mut TrainState)):
        self._on_epoch_end.push_back(h)

    fn add_on_batch_start(mut self, h: fn(mut TrainState, BatchInfo)):
        self._on_batch_start.push_back(h)

    fn add_on_batch_end(mut self, h: fn(mut TrainState, BatchInfo, Float64)):
        self._on_batch_end.push_back(h)

    fn add_on_validation_start(mut self, h: fn(mut TrainState)):
        self._on_validation_start.push_back(h)

    fn add_on_validation_end(mut self, h: fn(mut TrainState)):
        self._on_validation_end.push_back(h)

    fn add_on_test_start(mut self, h: fn(mut TrainState)):
        self._on_test_start.push_back(h)

    fn add_on_test_end(mut self, h: fn(mut TrainState)):
        self._on_test_end.push_back(h)

    fn add_on_lr_update(mut self, h: fn(mut TrainState, LRUpdateInfo)):
        self._on_lr_update.push_back(h)

    fn add_on_checkpoint_saved(mut self, h: fn(mut TrainState, String)):
        self._on_checkpoint_saved.push_back(h)

    # -----------------------------
    # Emission API (on_*)
    # Called by Trainer at appropriate times.
    # -----------------------------

    fn on_train_start(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_train_start.size()):
            self._on_train_start[i](state)
            i = i + 1

    fn on_train_end(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_train_end.size()):
            self._on_train_end[i](state)
            i = i + 1

    fn on_epoch_start(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_epoch_start.size()):
            self._on_epoch_start[i](state)
            i = i + 1

    fn on_epoch_end(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_epoch_end.size()):
            self._on_epoch_end[i](state)
            i = i + 1

    fn on_batch_start(mut self, mut state: TrainState, info: BatchInfo):
        var i = 0
        while i < Int(self._on_batch_start.size()):
            self._on_batch_start[i](state, info)
            i = i + 1

    fn on_batch_end(mut self, mut state: TrainState, info: BatchInfo, loss: Float64):
        var i = 0
        while i < Int(self._on_batch_end.size()):
            self._on_batch_end[i](state, info, loss)
            i = i + 1

    fn on_validation_start(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_validation_start.size()):
            self._on_validation_start[i](state)
            i = i + 1

    fn on_validation_end(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_validation_end.size()):
            self._on_validation_end[i](state)
            i = i + 1

    fn on_test_start(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_test_start.size()):
            self._on_test_start[i](state)
            i = i + 1

    fn on_test_end(mut self, mut state: TrainState):
        var i = 0
        while i < Int(self._on_test_end.size()):
            self._on_test_end[i](state)
            i = i + 1

    fn on_lr_update(mut self, mut state: TrainState, info: LRUpdateInfo):
        var i = 0
        while i < Int(self._on_lr_update.size()):
            self._on_lr_update[i](state, info)
            i = i + 1

    fn on_checkpoint_saved(mut self, mut state: TrainState, path: String):
        var i = 0
        while i < Int(self._on_checkpoint_saved.size()):
            self._on_checkpoint_saved[i](state, path)
            i = i + 1

    # -----------------------------
    # Utilities
    # -----------------------------

    fn clear(mut self):
        # Remove all handlers from all events
        self._on_train_start = List[fn(mut TrainState)]()
        self._on_train_end   = List[fn(mut TrainState)]()

        self._on_epoch_start = List[fn(mut TrainState)]()
        self._on_epoch_end   = List[fn(mut TrainState)]()

        self._on_batch_start = List[fn(mut TrainState, BatchInfo)]()
        self._on_batch_end   = List[fn(mut TrainState, BatchInfo, Float64)]()

        self._on_validation_start = List[fn(mut TrainState)]()
        self._on_validation_end   = List[fn(mut TrainState)]()
        self._on_test_start       = List[fn(mut TrainState)]()
        self._on_test_end         = List[fn(mut TrainState)]()

        self._on_lr_update        = List[fn(mut TrainState, LRUpdateInfo)]()
        self._on_checkpoint_saved = List[fn(mut TrainState, String)]()
