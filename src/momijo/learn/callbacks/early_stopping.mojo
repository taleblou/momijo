# Project:      Momijo
# Module:       learn.callbacks.early_stopping
# File:         callbacks/early_stopping.mojo
# Path:         src/momijo/learn/callbacks/early_stopping.mojo
#
# Description:  Early stopping callback for Momijo Learn. Monitors a scalar metric
#               (e.g., "val_loss" or "val_accuracy") and stops training when no
#               improvement is observed for a configured number of epochs. Supports
#               min/max modes, min_delta threshold, optional baseline, and restoring
#               the best model weights upon stop.
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
#   - Types: EarlyStopping
#   - Key methods: on_epoch_end(...), should_stop(), get_best(), reset(),
#                  capture_best(model), restore_best(model)
#   - Model contract: model must implement `state_dict() -> String`
#                     and `load_state_dict(state: String)`

from pathlib.path import Path
from collections.list import List

struct EarlyStopping:
    # Configuration
    var monitor: String
    var mode: String                 # "min" or "max"
    var patience: Int
    var min_delta: Float64
    var restore_best_weights: Bool
    var verbose: Bool
    var baseline: Optional[Float64]

    # Runtime state
    var best_score: Float64
    var best_epoch: Int
    var wait: Int
    var stopped_epoch: Int
    var best_state: Optional[String]  # serialized state_dict()

    fn __init__(
        out self,
        monitor: String = String("val_loss"),
        mode: String = String("min"),
        patience: Int = 10,
        min_delta: Float64 = 0.0,
        baseline: Optional[Float64] = None,
        restore_best_weights: Bool = True,
        verbose: Bool = False
    ):
        self.monitor = monitor
        # normalize mode
        var m = mode
        if not (m == String("min") or m == String("max")):
            m = String("min")
        self.mode = m

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.baseline = baseline

        # Initialize runtime state
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.best_state = None

        # Initialize best_score based on mode and baseline
        if self.baseline is None:
            if self.mode == String("min"):
                # use a large positive number as +infinity sentinel
                self.best_score = 1.0e300
            else:
                # large negative as -infinity sentinel
                self.best_score = -1.0e300
        else:
            self.best_score = self.baseline.value()

    # Reset internal counters and best scores (does not change configuration)
    fn reset(mut self):
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.best_state = None
        if self.baseline is None:
            if self.mode == String("min"):
                self.best_score = 1.0e300
            else:
                self.best_score = -1.0e300
        else:
            self.best_score = self.baseline.value()

    # Return true if "current" is an improvement over "best_score" given mode/min_delta.
    fn _is_improved(self, current: Float64) -> Bool:
        if self.mode == String("min"):
            # Need a decrease of at least min_delta
            return current < (self.best_score - self.min_delta)
        # mode == "max"
        return current > (self.best_score + self.min_delta)

    # Optionally capture model weights as best_state. Requires model with state_dict().
    fn capture_best(mut self, model):
        if self.restore_best_weights:
            var s = model.state_dict()
            self.best_state = Optional[String](s)

    # Optionally restore the best_state back to model. Requires model with load_state_dict(...).
    fn restore_best(self, model):
        if self.restore_best_weights:
            if not (self.best_state is None):
                model.load_state_dict(self.best_state.value())

    # Call this at the end of each epoch with the monitored value.
    # Returns true if training should stop.
    fn on_epoch_end(mut self, epoch: Int, current_value: Float64, model = None) -> Bool:
        if self._is_improved(current_value):
            self.best_score = current_value
            self.best_epoch = epoch
            self.wait = 0
            if not (model is None):
                self.capture_best(model)
            if self.verbose:
                print(String("[EarlyStopping] Improved ") + self.monitor + String(" = ") + String(self.best_score))
            return False

        # No improvement
        self.wait = self.wait + 1
        if self.verbose:
            print(String("[EarlyStopping] No improvement for ") + String(self.wait) + String(" epoch(s)"))

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if not (model is None):
                self.restore_best(model)
            if self.verbose:
                print(String("[EarlyStopping] Stopping at epoch ") + String(epoch)
                      + String("; best epoch=") + String(self.best_epoch)
                      + String(", best ") + self.monitor + String("=") + String(self.best_score))
            return True

        return False

    # Convenience alias
    fn should_stop(self) -> Bool:
        return self.stopped_epoch >= 0

    # Return best score and epoch observed so far.
    fn get_best(self) -> (Float64, Int):
        return (self.best_score, self.best_epoch)

    # Minimal JSON-like summary string (for logging/debug).
    fn summary(self) -> String:
        var s = String("{ 'monitor': '") + self.monitor + String("', 'mode': '") + self.mode + String("'")
        s = s + String(", 'patience': ") + String(self.patience)
        s = s + String(", 'min_delta': ") + String(self.min_delta)
        s = s + String(", 'best_score': ") + String(self.best_score)
        s = s + String(", 'best_epoch': ") + String(self.best_epoch)
        s = s + String(", 'stopped_epoch': ") + String(self.stopped_epoch) + String(" }")
        return s
